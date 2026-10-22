"""The last lever: fewer integration steps in the flow head. Judge by ear.

WHY THIS ONE IS DIFFERENT FROM EVERY OTHER CHANGE HERE

The three accelerations in `faster_ming` leave the arithmetic alone, so the
waveform is identical and a bench can settle them. This one does not. The flow
head integrates in 10 steps by default, and the head is still two thirds of
what a decode step costs after grafting, so halving the steps nearly halves the
remaining time -- at the price of a different waveform.

Whether that waveform is *worse* is not something a number can answer. An
embedding distance would say "different" for a hundredth of a second of shift,
and a transcription would not notice a mangled word at all. So this bench does
not score anything. It renders the same sentences at each step count, prints
the speed, and writes the files out in a numbered folder for listening.

Upstream's sampler has pruned timestep tables for 5, 6, 7, 10, 12 and 16 steps
(`get_epss_timesteps`); anything else falls back to a linear schedule, which is
a different method and not a fair comparison, so only the tabled values are
offered.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

TABLED = (5, 6, 7, 10, 12, 16)
SENTENCES = [
    ("fr", "Il y a trois boites dans le couloir, et la fenetre du salon est "
           "restee ouverte toute la nuit."),
    ("en", "There are three boxes in the hallway, and the living room window "
           "stayed open all night."),
    ("fr-court", "Non. Pas comme ca."),
]


def write_wav(path, x, rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ming", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True, help="folder for the wav files")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 7, 5],
                    help=f"step counts to render; tabled values are {TABLED}")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    for n in args.steps:
        if n not in TABLED:
            ap.error(f"{n} has no pruned timestep table; pick from {TABLED}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, args.ming)
    os.chdir(args.ming)

    import torch
    from loguru import logger
    logger.remove()
    import tqdm
    real = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = lambda self, *a, **k: real(
        self, *a, **{**k, "disable": True})

    import faster_ming
    from ming_tts import MingTTS

    tts = MingTTS(args.weights, device="cuda:0")
    print(f"{torch.cuda.get_device_name(0)} - torch {torch.__version__}")
    print(f"\n{'steps':>6} {'lang':>9} {'audio':>7} {'compute':>8} "
          f"{'x real':>7}")

    graft = None
    for n in args.steps:
        if graft is not None:
            graft.undo()
        graft = faster_ming.accelerate(tts.model, steps=n)
        for name, text in SENTENCES:
            lang = name.split("-")[0]
            torch.manual_seed(args.seed)
            # One throwaway render per (steps, prompt) so the capture does not
            # land inside the timed one.
            tts.synthesize(task="TTS", text=text, language_id=lang,
                           prompt_wav_path=args.reference, use_spk_emb=True,
                           raw_text=True)
            torch.manual_seed(args.seed)
            start = time.perf_counter()
            wav = tts.synthesize(task="TTS", text=text, language_id=lang,
                                 prompt_wav_path=args.reference,
                                 use_spk_emb=True, raw_text=True)
            torch.cuda.synchronize()
            took = time.perf_counter() - start
            x = wav.float().reshape(-1).cpu().numpy()
            seconds = x.size / tts.sample_rate
            write_wav(os.path.join(args.out, f"{name}-{n:02d}pas.wav"), x,
                      tts.sample_rate)
            print(f"{n:6d} {name:>9} {seconds:6.2f}s {took:7.2f}s "
                  f"{seconds / took:6.2f}x", flush=True)

    print(f"\nfiles in {args.out}")
    print("Listen before you decide: at 10 steps the waveform matches "
          "upstream to the bit, below that it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
