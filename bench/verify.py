"""Does it still say exactly the same thing, and how much faster?

THE ORDER OF THESE TWO QUESTIONS IS NOT NEGOTIABLE

A faster voice engine that does not produce the same waveform is a different
model, not an optimisation. So identity is checked first, to the bit, and only
then the clock.

Seven cases, chosen because each one stresses a different stage:

    short            a two-second utterance, where a weak graph shows up fast
    long             many steps, so the static cache approaches its 512 slots
    split text       `raw_text=False`, which re-enters `generate` per fragment
                     and is the only path where cache state could survive from
                     one fragment to the next
    two emotions     a second guidance scale, hence a second captured graph:
                     the first version of this package shared buffers between
                     graphs, which is precisely the bug it had
    english          a different language id
    short again      proves the cache resets cleanly between utterances

`cache_prompts=False` on purpose. The prompt cache is the right thing in
production, but it makes the *first* render consume the VAE posterior's random
draws and the second not, so the two paths would start from offset seeds and
the comparison would mean nothing. Measure the floor before believing a delta.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

SHORT = "Non. Pas comme ca."
LONG = (
    "Il y a trois boites dans le couloir, et la fenetre du salon est restee "
    "ouverte toute la nuit, ce qui explique sans doute pourquoi le parquet "
    "est humide ce matin, et pourquoi le chat a decide de dormir sur la "
    "machine a laver plutot que dans son panier habituel."
)
MEDIUM = ("La page n'est pas a jour, je vais chercher ailleurs, "
          "ca ne prendra pas longtemps.")

CASES = [
    ("short fr", dict(task="TTS", text=SHORT, language_id="fr",
                      raw_text=True)),
    ("long fr", dict(task="TTS", text=LONG, language_id="fr", raw_text=True)),
    ("long, split", dict(task="TTS", text=LONG, language_id="fr",
                         raw_text=False)),
    ("anger, cfg 3", dict(task="EMOTION", text=MEDIUM, language_id="fr",
                          controls={"emotion": "愤怒"}, cfg=3.0,
                          raw_text=True)),
    ("joy, cfg 3", dict(task="EMOTION", text=MEDIUM, language_id="fr",
                        controls={"emotion": "高兴"}, cfg=3.0,
                        raw_text=True)),
    ("english, cfg 2", dict(task="TTS", text="I don't know, maybe it's "
                                             "simpler than that.",
                            language_id="en", raw_text=True)),
    ("short fr again", dict(task="TTS", text=SHORT, language_id="fr",
                            raw_text=True)),
]


def quiet():
    """Upstream prints a progress bar and a prompt dump per utterance."""
    import tqdm
    from loguru import logger

    logger.remove()
    real = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = lambda self, *a, **k: real(
        self, *a, **{**k, "disable": True})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ming", required=True,
                    help="path to the inclusionAI/Ming-3.0-tts inference repo")
    ap.add_argument("--weights", required=True,
                    help="path to the Ming-3.0-tts-0.5B checkpoint")
    ap.add_argument("--reference", required=True,
                    help="a wav file to clone from")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Resolved before the chdir: upstream's modules use relative paths for
    # their own data files, so the bench has to run from the Ming repo root.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, args.ming)
    os.chdir(args.ming)

    import torch

    quiet()
    import faster_ming
    from ming_tts import MingTTS

    tts = MingTTS(args.weights, device="cuda:0", cache_prompts=False)
    model = tts.model
    print(f"{torch.cuda.get_device_name(0)} - torch {torch.__version__} - "
          f"faster-ming {faster_ming.__version__}")

    def render(kw):
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        wav = tts.synthesize(prompt_wav_path=args.reference, use_spk_emb=True,
                             **kw)
        torch.cuda.synchronize()
        return (wav.float().reshape(-1).cpu().numpy(),
                time.perf_counter() - start)

    torch.cuda.reset_peak_memory_stats()
    print("\nupstream", flush=True)
    before = {}
    for name, kw in CASES:
        wav, took = render(kw)
        before[name] = (wav, took)
        print(f"  {name:16} {wav.size / tts.sample_rate:6.2f} s in "
              f"{took:6.2f} s ({wav.size / tts.sample_rate / took:5.2f}x)",
              flush=True)
    peak_before = torch.cuda.max_memory_allocated() / 2 ** 20

    graft = faster_ming.accelerate(model)
    torch.cuda.reset_peak_memory_stats()
    # One throwaway render: it triggers the remaining captures and takes the
    # RNG stream out of the equation. Without it a capture would land in the
    # middle of a measured render and shift every draw after it.
    render(CASES[0][1])
    if graft.failures:
        print(f"\n  CAPTURE FAILURES: {graft.failures}")

    print("\nfaster-ming", flush=True)
    gains, exact = [], 0
    for name, kw in CASES:
        wav, took = render(kw)
        ref, ref_took = before[name]
        same = ref.size == wav.size and bool(np.array_equal(ref, wav))
        exact += int(same)
        gains.append(ref_took / took)
        note = "bit-identical" if same else "DIFFERENT"
        if not same:
            note += (f" ({ref.size} vs {wav.size} samples)"
                     if ref.size != wav.size
                     else f" (max delta {np.abs(ref - wav).max():.3e})")
        print(f"  {name:16} {wav.size / tts.sample_rate:6.2f} s in "
              f"{took:6.2f} s ({wav.size / tts.sample_rate / took:5.2f}x)"
              f"  {ref_took / took:5.2f}x  {note}", flush=True)
    peak_after = torch.cuda.max_memory_allocated() / 2 ** 20

    print(f"\n  {exact}/{len(CASES)} bit-identical")
    print(f"  median {sorted(gains)[len(gains) // 2]:.2f}x "
          f"(min {min(gains):.2f}x, max {max(gains):.2f}x)")
    print(f"  {graft.flow_head.captures} flow-head capture(s), one per "
          f"guidance setting; {graft.backbone.prefills} prefills, "
          f"{graft.backbone.replays} graphed decode steps")
    print(f"  peak memory {peak_before:.0f} MB -> {peak_after:.0f} MB "
          f"({peak_after - peak_before:+.0f} MB)")

    graft.undo()
    wav, took = render(CASES[0][1])
    ref, ref_took = before["short fr"]
    print(f"\n  after undo(): {took:.2f} s (upstream {ref_took:.2f} s), "
          f"waveform restored "
          f"{ref.size == wav.size and bool(np.array_equal(ref, wav))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
