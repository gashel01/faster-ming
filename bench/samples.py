"""Render the listening samples in `samples/`, and prove they are the same.

Each sentence is rendered twice, upstream then grafted, with the same seed.
Only the grafted file is written, because the two are compared to the bit
before anything touches the disk: if they differ, the script says so and
writes nothing for that sentence. `samples/README.md` is regenerated with the
timings and the SHA-256 of *both* renders, so anyone can check the claim with
the files in hand rather than take it from a table.

USE A REFERENCE YOU ARE ALLOWED TO REDISTRIBUTE

The samples clone the reference voice, and they get published. Upstream ships
example prompts under its MIT licence in `data/wavs/` of the inference repo,
which is what these samples are meant to be made from:

    python bench/samples.py --ming /path/to/Ming-3.0-tts \\
        --weights /path/to/Ming-3.0-tts-0.5B \\
        --reference data/wavs/<one of upstream's prompts>.wav

`--reference` is resolved inside the Ming repo when it is a relative path,
since upstream's modules need that repo as the working directory anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import wave

import numpy as np

SENTENCES = [
    ("fr", dict(task="TTS", language_id="fr", raw_text=True,
                text="Il y a trois boîtes dans le couloir, et la fenêtre du "
                     "salon est restée ouverte toute la nuit.")),
    ("en", dict(task="TTS", language_id="en", raw_text=True,
                text="I don't know, maybe it's simpler than that. Let me "
                     "check again before we decide.")),
    ("zh", dict(task="TTS", language_id="zh", raw_text=True,
                text="今天的天气很好，我们一起去公园散步吧。")),
    # Emotion control is weak in French on this model, so the guided samples
    # are in English and in Chinese, the language it was mostly trained on.
    ("en-anger", dict(task="EMOTION", language_id="en", raw_text=True,
                      controls={"emotion": "愤怒"}, cfg=3.0,
                      text="Are you kidding me? I told you three times not "
                           "to touch that!")),
    ("zh-anger", dict(task="EMOTION", language_id="zh", raw_text=True,
                      controls={"emotion": "愤怒"}, cfg=3.0,
                      text="你到底在干什么？我已经跟你说了三遍，别碰那个！")),
]


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    """16-bit PCM through the standard library, so the bench needs nothing
    beyond what upstream already installs."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def digest(samples: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ming", required=True,
                    help="path to the inclusionAI/Ming-3.0-tts inference repo")
    ap.add_argument("--weights", required=True,
                    help="path to the Ming-3.0-tts-0.5B checkpoint")
    ap.add_argument("--reference", required=True,
                    help="a wav you may redistribute; relative paths are "
                         "resolved inside the Ming repo")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.abspath(args.weights)
    sys.path.insert(0, root)
    sys.path.insert(0, args.ming)
    os.chdir(args.ming)
    reference = os.path.abspath(args.reference)

    import torch

    sys.path.insert(0, os.path.join(root, "bench"))
    from verify import quiet

    quiet()
    import faster_ming
    from ming_tts import MingTTS

    tts = MingTTS(weights, device="cuda:0", cache_prompts=False)
    rate = tts.sample_rate
    device = torch.cuda.get_device_name(0)

    def render(kw):
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        wav = tts.synthesize(prompt_wav_path=reference, use_spk_emb=True,
                             **kw)
        torch.cuda.synchronize()
        return (wav.float().reshape(-1).cpu().numpy(),
                time.perf_counter() - start)

    before = {name: render(kw) for name, kw in SENTENCES}

    graft = faster_ming.accelerate(tts.model)
    # Same throwaway render as verify.py: the remaining captures happen here
    # rather than in the middle of a measured render.
    render(SENTENCES[0][1])
    render(SENTENCES[-1][1])

    rows, written = [], 0
    for name, kw in SENTENCES:
        wav, took = render(kw)
        ref, ref_took = before[name]
        same = ref.size == wav.size and bool(np.array_equal(ref, wav))
        seconds = wav.size / rate
        if same:
            write_wav(os.path.join(out_dir, f"{name}.wav"), wav, rate)
            written += 1
        rows.append((name, seconds, ref_took, took, same,
                     digest(ref), digest(wav)))
        print(f"  {name:9} {seconds:5.2f} s audio  upstream {ref_took:5.2f} s"
              f"  faster-ming {took:5.2f} s  "
              f"{'bit-identical' if same else 'DIFFERENT, not written'}",
              flush=True)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Samples\n\n")
        f.write(f"Rendered by `bench/samples.py` on {device}, torch "
                f"{torch.__version__}, faster-ming {faster_ming.__version__}, "
                f"seed {args.seed}. Reference voice: `"
                f"{os.path.relpath(reference, args.ming).replace(os.sep, '/')}`"
                " from the upstream inference repo.\n\n")
        f.write("Each file is the faster-ming render. The upstream render of "
                "the same sentence was compared to it sample by sample before "
                "it was written; both hashes are below, over the float32 "
                "samples before 16-bit conversion.\n\n")
        f.write("| file | audio | upstream | faster-ming | speedup | "
                "identical | SHA-256 upstream | SHA-256 faster-ming |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name, seconds, ref_took, took, same, h_ref, h_new in rows:
            link = f"[{name}.wav]({name}.wav)" if same else f"{name} (not written)"
            f.write(f"| {link} | {seconds:.2f} s | {ref_took:.2f} s | "
                    f"{took:.2f} s | {ref_took / took:.2f}x | "
                    f"{'yes' if same else '**no**'} | `{h_ref[:16]}` | "
                    f"`{h_new[:16]}` |\n")

    print(f"\n  {written}/{len(SENTENCES)} written to {out_dir}")
    return 0 if written == len(SENTENCES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
