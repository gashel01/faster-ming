"""Where do the milliseconds of one decode step go?

THE NUMBER THAT MAKES THE REST READABLE

This bench starts by timing a trivial kernel launch. Without that floor, "a
step costs 229 ms" does not say whether the machine is compute-bound or
dispatch-bound, and that is the whole difference between reaching for
quantization and reaching for CUDA graphs. On Linux a trivial launch costs
5-10 microseconds; on Windows, under the WDDM driver model, 13-18. Multiply by
the few thousand kernels a step dispatches and the 229 ms is accounted for.

It then times each component in isolation, with CUDA events, after warmup, and
reports the median rather than the mean -- one driver hiccup ruins a mean over
thirty samples.

One decode step produces exactly one latent frame: `patch_size: 1` in the
config, and the audio tokenizer's `hop_size: 882` with its own `patch_size: 4`
makes that 3528 samples at 44.1 kHz, or 80 ms of audio. So the real-time
threshold is 12.5 steps per second, i.e. 80 ms per step, and that is printed
alongside so every row can be read against the target.
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys


def timed(fn, n=30, warmup=5):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return st.median(samples)


def launch_floor():
    """Cost of one trivial kernel, launched from Python, on this card."""
    import torch

    x = torch.zeros(1, device="cuda")
    n = 2000

    def burst():
        for _ in range(n):
            x.add_(1.0)

    return timed(burst, n=5, warmup=2) / n * 1000  # microseconds per kernel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ming", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--accelerate", action="store_true",
                    help="profile the grafted path instead of upstream")
    args = ap.parse_args()

    # Resolved before the chdir: upstream's modules use relative paths for
    # their own data files, so the bench has to run from the Ming repo root.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, args.ming)
    os.chdir(args.ming)

    import torch
    from loguru import logger
    logger.remove()

    import tts_tasks
    from ming_tts import MingTTS

    print(f"torch {torch.__version__} - {torch.cuda.get_device_name(0)}")
    floor = launch_floor()
    # No pass/fail verdict here: the number is the point. Linux on a
    # datacenter card lands around 5-10 microseconds, Windows under WDDM
    # around 13-18, and a step dispatches a few thousand kernels -- so this
    # one figure is what decides whether the bottleneck is arithmetic or
    # dispatch.
    print(f"launch floor: {floor:.1f} microseconds per trivial kernel "
          f"(Linux is typically 5-10, Windows/WDDM 13-18)")

    tts = MingTTS(args.weights, device="cuda:0")
    model = tts.model
    if args.accelerate:
        import faster_ming
        faster_ming.accelerate(model)
        print("grafted")

    hop = model.audio.encoder.hop_size
    tpatch = model.audio.encoder.patch_size
    seconds = hop * tpatch / tts.sample_rate
    print(f"one latent frame = {hop}x{tpatch} = {hop * tpatch} samples "
          f"= {seconds * 1000:.0f} ms of audio")
    print(f"-> real time needs {1 / seconds:.1f} steps/s, "
          f"{seconds * 1000:.0f} ms per step")

    text = "Il y a trois boites dans le couloir, et la fenetre est ouverte."
    prompt_latent, spk_emb = tts.prepare_prompt(args.reference, True)
    request = tts_tasks.build_task_generation_kwargs(
        "TTS", text=text, language_id="fr", prompt_wav_path=args.reference,
        use_spk_emb=True, instruction=None, controls=None,
        instruction_schema=None, prompt_text=None, spk_emb_path=None,
        use_zero_spk_emb=False, cfg=2.0, sigma=0.25, temperature=0.0)

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        input_ids, embeds = model.prepare_input_embed(
            prompt=request["prompt"], text=text,
            spk_emb=[e.clone() for e in spk_emb],
            instruction=request["instruction"], prompt_latent=prompt_latent,
            prompt_text=request["prompt_text"])
        print(f"prompt is {embeds.shape[1]} tokens "
              f"(decode budget {512 - embeds.shape[1]} steps)")

        mask = torch.ones(input_ids.shape, device="cuda")
        positions = mask.cumsum(-1) - 1
        out = model.model(attention_mask=mask, position_ids=positions,
                          inputs_embeds=embeds, use_cache=True,
                          output_hidden_states=True, return_dict=True)
        cache = out.past_key_values
        z = out.hidden_states[-1][:, -1:, :]
        hist = torch.zeros(1, model.history_patch_size, model.latent_dim,
                           device="cuda", dtype=z.dtype)
        one = torch.ones(1, 1, device="cuda")
        next_pos = positions[:, -1:] + 1
        zero = torch.zeros(1, 1, model.latent_dim, device="cuda",
                           dtype=z.dtype)
        embed1 = model.linear_proj_audio(zero)

        rows = [
            ("backbone, 1 token",
             lambda: model.model(attention_mask=one, position_ids=next_pos,
                                 inputs_embeds=embed1, use_cache=True,
                                 output_hidden_states=True, return_dict=True,
                                 past_key_values=cache)),
            ("flow head, 10 steps",
             lambda: model.flowloss.sample(z, hist, 2.0, model.patch_size,
                                           sigma=0.25, temperature=0.0)),
            ("aggregator", lambda: model.linear_proj_audio(zero)),
            ("stop head + host sync",
             lambda: bool(model.stop_head(z)[0][0].softmax(-1)[1] > 0.5)),
        ]
        print(f"\n{'component':26} {'ms':>8} {'% of step':>10}")
        times = [(name, timed(fn)) for name, fn in rows]
        total = sum(ms for _, ms in times)
        for name, ms in times:
            print(f"{name:26} {ms:8.2f} {100 * ms / total:9.0f}%")
        print(f"{'sum of one step':26} {total:8.2f}")
        print(f"   {1000 / total:.1f} steps/s -> "
              f"{seconds * 1000 / total:.2f}x real time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
