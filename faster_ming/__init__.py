"""faster-ming: Ming-3.0-tts, same weights, same output, about 7x the speed.

    from ming_tts import MingTTS
    from faster_ming import accelerate

    tts = MingTTS("inclusionAI/Ming-3.0-tts-0.5B", device="cuda:0")
    accelerate(tts.model)          # that is the whole integration

    wav = tts.synthesize(task="TTS", text="...", language_id="fr",
                         prompt_wav_path="reference.wav", use_spk_emb=True)

Nothing here retrains, quantizes or rewrites the model. Three changes, all of
which leave the arithmetic alone:

    flow-matching head in a CUDA graph    167.06 ms -> 19.77 ms  (75% of a step)
    vocabulary head muted                  44.5 ms  -> 38.1 ms    (unused logits)
    decode step graphed on a static cache  41.06 ms ->  7.88 ms
    aggregator in a CUDA graph             15.76 ms ->  1.91 ms
    ------------------------------------------------------------------------
    one decode step, 80 ms of audio       224.14 ms -> 29.80 ms
                                            0.36x   ->  2.68x real time

Measured end to end on an RTX 3070 Ti Laptop (8 GB, Windows/WDDM), bfloat16,
seven cases covering short, long, split-text, two guidance scales and two
languages: **about 7x (5.4-8.1x)**, from 0.33x to 1.8-2.8x real time, **output
identical to the bit in all seven**, +207 MB peak memory.

Bit identity is the point, not a bonus. A faster voice engine that does not
produce the same waveform is a different model, so the benches check the
waveform before they look at the clock. It holds because none of the three
changes touches the computation: a graph replays the same kernels on the same
weights and only changes the dispatch, the muted head drops a product whose
result was discarded, and a static cache adds exact zeros to a softmax sum.

Requires: the upstream inference repo (github.com/inclusionAI/Ming-3.0-tts) on
the import path, torch with CUDA, transformers >= 4.46 and < 4.54 (verified on 4.46.3).
"""

from __future__ import annotations

from .backbone import GraphedBackbone, MutedLMHead, mute_vocabulary
from .capture import Captured, GraphedCall, capture
from .flow import GraphedFlowHead

__version__ = "0.1.0"


class Graft:
    """What was replaced, and how to put it all back.

    The graft only ever touches entry points, never upstream's code:
    `flowloss.sample`, `linear_proj_audio.forward`, `lm_head` and
    `model.forward`. So it undoes cleanly, and a bench can compare both paths
    inside one process on one card in one state -- which is the only way to
    prove the output did not move.
    """

    def __init__(self, model, flow_head, aggregator, real_lm_head, backbone):
        self.model = model
        self.flow_head = flow_head
        self.aggregator = aggregator
        self.backbone = backbone
        self._real_lm_head = real_lm_head

    @property
    def replays(self) -> int:
        return (self.flow_head.replays
                + (self.aggregator.replays if self.aggregator else 0)
                + (self.backbone.replays if self.backbone else 0))

    @property
    def failures(self) -> list:
        return (self.flow_head.failures
                + (self.aggregator.failures if self.aggregator else [])
                + (self.backbone.failures if self.backbone else []))

    def undo(self) -> None:
        if self._real_lm_head is not None:
            self.model.model.lm_head = self._real_lm_head
        self.model.flowloss.sample = self.flow_head.original
        if self.aggregator is not None:
            # Dropping the instance attribute lets the class method take over.
            del self.model.linear_proj_audio.forward
        if self.backbone is not None:
            del self.model.model.forward


def accelerate(model, steps: int = 10, aggregator: bool = True,
               mute: bool = True, backbone: bool = True) -> Graft:
    """Graft the accelerations onto an already-loaded Ming model.

    `model` is the `BailingMMNativeForConditionalGeneration` behind
    `MingTTS.model`. Each stage can be turned off to isolate it in a bench.
    Returns a `Graft`; call `undo()` to restore every entry point.
    """
    flow_head = GraphedFlowHead(
        model.flowloss,
        patch_size=model.patch_size,
        latent_dim=model.latent_dim,
        llm_dim=model.model.config.hidden_size,
        device=model.device,
        steps=steps,
    )
    model.flowloss.sample = flow_head

    graphed_aggregator = None
    if aggregator:
        graphed_aggregator = GraphedCall(
            model.linear_proj_audio.forward,
            (1, model.patch_size, model.latent_dim),
            model.device,
        )
        model.linear_proj_audio.forward = graphed_aggregator

    real_lm_head = mute_vocabulary(model) if mute else None

    graphed_backbone = None
    if backbone:
        graphed_backbone = GraphedBackbone(model.model, model.device)
        graphed_backbone.prepare()
        model.model.forward = graphed_backbone

    return Graft(model, flow_head, graphed_aggregator, real_lm_head,
                 graphed_backbone)


__all__ = ["accelerate", "Graft", "GraphedFlowHead", "GraphedBackbone",
           "GraphedCall", "Captured", "capture", "MutedLMHead",
           "mute_vocabulary", "__version__"]
