"""The Qwen2.5 backbone: a muted vocabulary head, and a graphed decode step.

TWO SEPARATE WINS, AND THE SMALL ONE IS FREE

`BailingMMNativeForConditionalGeneration.sample` reads
`outputs.hidden_states[-1]` and never looks at the logits. Yet `lm_head` is
896 x 151936: 272 MB of weights re-read on every decode step, the largest
tensor in the model, computed and thrown away. Returning a one-channel view
instead costs no kernel at all and takes a step from 44.5 ms to 38.1 ms.

The rest of the backbone's time is not arithmetic either. A single-token pass
through 24 layers is a few hundred kernels, perhaps 9 ms of dispatch on this
machine -- the other 29 ms is the Python above it: the legacy tuple cache
converted both ways on every call, 48 `torch.cat` calls growing the KV cache,
and the attention-mask logic. A CUDA graph removes all of it, and a graph needs
a cache whose shape never moves.

WHAT THE STATIC CACHE CHANGES, AND WHAT MEASUREMENT SAID

A dynamic cache shows attention exactly the N filled positions. A static cache
shows 512 and masks the rest to negative infinity. In exact arithmetic that is
the same thing, since masked positions contribute zero. In floating point it
was not guaranteed: the kernel tiles 512 keys instead of N, so the
accumulation order could change.

It does not. Output is identical to the bit across seven cases -- short, long,
split-text, two guidance settings, two languages -- because adding exact zeros
to a sum never moves it, in any order. The prediction was wrong and the
measurement is what counts.

Prefill still runs on the original path. It happens once per utterance so
speeding it up buys nothing, and leaving it untouched removes a source of
divergence for free; its keys and values are then transplanted into the static
cache.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture

# Upstream decodes with a 512-position cache, so an utterance can run for at
# most (512 - prompt length) steps. Keep these in step with
# `modeling_bailingmm.MAX_CACHE_LEN`.
MAX_CACHE_LEN = 512


class MutedLMHead(torch.nn.Module):
    """A vocabulary projection that projects nothing.

    Returns a single-channel view: no kernel, no copy, and the caller cannot
    tell because it does not look. `sample_text` and `generate_text` *do* read
    the logits, so the graft keeps the real head aside and `undo()` puts it
    back.
    """

    def forward(self, x):
        return x[..., :1]


def mute_vocabulary(model):
    """Swap in the muted head and return the real one, for undoing."""
    real = model.model.lm_head
    model.model.lm_head = MutedLMHead()
    return real


class GraphedBackbone:
    """A graphed, static-cache decode step for the Qwen2.5 backbone.

    Installed over `Qwen2ForCausalLM.forward`, it keeps its own cache and
    position counter, so it owns the whole utterance: a call with
    `past_key_values=None` means a new utterance and resets both.

    Batch size 1 only, which is what the upstream sampler does anyway
    ("this implementation currently assumes a batch size of 1").
    """

    def __init__(self, causal, device, dtype=torch.bfloat16,
                 max_cache=MAX_CACHE_LEN):
        from transformers import StaticCache

        self.call = causal.forward
        self.body = causal.model
        self.config = causal.config
        self.device = device
        self.dtype = dtype
        self.max_cache = max_cache
        self.cache = StaticCache(
            config=causal.config, batch_size=1, max_cache_len=max_cache,
            device=device, dtype=dtype,
        )
        self.position = 0
        self.captured = None
        self.failures: list[str] = []
        self.replays = 0
        self.prefills = 0

    def prepare(self) -> bool:
        """Capture now, while the cache is still empty.

        Deliberately not lazy. Capture writes warmup keys into the cache and
        has to clear them, so it cannot happen on the first decode step, which
        comes right after a prefill that must survive. Doing it at graft time
        means there is nothing to lose.
        """
        try:
            self.captured = self._build()
        except Exception as exc:  # pragma: no cover - driver dependent
            self.failures.append(repr(exc))
            self.captured = False
        return self.captured is not False

    def _build(self) -> Captured:
        embeds = torch.zeros(1, 1, self.config.hidden_size,
                             device=self.device, dtype=self.dtype)
        # The upstream sampler derives positions from a float mask, so
        # `position_ids` arrives as float32. Kept as-is rather than fixed:
        # the rotary embedding reads that value exactly as it comes.
        positions = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
        cursor = torch.zeros(1, device=self.device, dtype=torch.long)

        def body():
            return self.body(
                inputs_embeds=embeds, attention_mask=None,
                position_ids=positions, past_key_values=self.cache,
                use_cache=True, cache_position=cursor, return_dict=True,
            ).last_hidden_state

        graph, output, context = capture(body)
        # Warmup and capture both wrote at position 0; clear so the first
        # utterance does not decode on top of warmup keys.
        self.cache.reset()
        return Captured(graph=graph, output=output,
                        inputs={"embeds": embeds, "positions": positions},
                        held={"cursor": cursor, "context": context})

    def _transplant(self, legacy, n):
        """Copy the prefill's keys and values into the static cache."""
        for i, (k, v) in enumerate(legacy):
            self.cache.key_cache[i][:, :, :n].copy_(k)
            self.cache.value_cache[i][:, :, :n].copy_(v)

    def _eager(self, embeds, position_ids):
        cursor = torch.arange(self.position, self.position + embeds.shape[1],
                              device=self.device)
        out = self.body(
            inputs_embeds=embeds, attention_mask=None,
            position_ids=position_ids, past_key_values=self.cache,
            use_cache=True, cache_position=cursor, return_dict=True)
        self.position += embeds.shape[1]
        return BackboneOutput(out.last_hidden_state, self.cache)

    def __call__(self, **kw):
        embeds = kw["inputs_embeds"]
        if kw.get("past_key_values") is None:
            # New utterance: clear, then prefill on the untouched original
            # path and transplant the result.
            self.cache.reset()
            self.position = 0
            out = self.call(**kw)
            n = embeds.shape[1]
            self._transplant(out.past_key_values, n)
            self.position = n
            self.prefills += 1
            return BackboneOutput(out.hidden_states[-1], self.cache)

        if self.captured is False or embeds.shape[1] != 1:
            return self._eager(embeds, kw["position_ids"])

        self.captured.held["cursor"].fill_(self.position)
        last = self.captured.replay(embeds=embeds,
                                    positions=kw["position_ids"])
        self.position += 1
        self.replays += 1
        return BackboneOutput(last, self.cache)


class BackboneOutput:
    """The two attributes the upstream sampler reads. Nothing else is needed.

    `hidden_states` holds one element because the caller only ever takes index
    -1, and for `Qwen2Model` that element is the output *after* the final norm,
    which is exactly `last_hidden_state`. The substitution is exact.
    """

    __slots__ = ("hidden_states", "past_key_values")

    def __init__(self, last, cache):
        self.hidden_states = (last,)
        self.past_key_values = cache


__all__ = ["GraphedBackbone", "BackboneOutput", "MutedLMHead",
           "mute_vocabulary", "MAX_CACHE_LEN"]
