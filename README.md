# faster-ming

[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) inference, **about 7x faster**, with a waveform **identical to the bit**.

No retraining, no quantization, no distillation, no rewritten model. Three changes, all of
which leave the arithmetic alone.

```python
from ming_tts import MingTTS
from faster_ming import accelerate

tts = MingTTS("inclusionAI/Ming-3.0-tts-0.5B", device="cuda:0")
accelerate(tts.model)          # that is the whole integration

wav = tts.synthesize(task="TTS", text="...", language_id="fr",
                     prompt_wav_path="reference.wav", use_spk_emb=True)
```

## Measured

RTX 3070 Ti Laptop (8 GB, Windows / WDDM), bfloat16, seed fixed, prompt cache off so both
paths consume the same random stream.

| case | audio | upstream | faster-ming | speedup | waveform |
|---|---|---|---|---|---|
| short, fr | 1.60 s | 5.49 s (0.29x) | 0.79 s (2.03x) | 6.97x | identical |
| long, fr | 13.40 s | 38.60 s (0.35x) | 4.78 s (2.80x) | 8.07x | identical |
| long, split text | 15.30 s | 44.03 s (0.35x) | 5.65 s (2.71x) | 7.80x | identical |
| anger, cfg 3 | 4.70 s | 13.88 s (0.34x) | 2.57 s (1.83x)¹ | 5.39x | identical |
| joy, cfg 3 | 4.70 s | 13.88 s (0.34x) | 1.84 s (2.55x) | 7.54x | identical |
| english, cfg 2 | 2.10 s | 6.17 s (0.34x) | 0.92 s (2.28x) | 6.71x | identical |
| short, fr, again | 1.60 s | 5.06 s (0.32x) | 0.79 s (2.03x) | 6.43x | identical |

**7/7 bit-identical. Median 6.97x (5.39-8.07x). 0.33x -> 1.8-2.8x real time.
Peak memory 2916 MB -> 3123 MB (+207 MB).**

¹ includes the one-off capture of a second graph, the first time a new guidance scale is
used (~0.8 s).

Two runs of the same bench gave medians of 6.97x and 7.25x, so read the figure as "about
seven", not as three significant digits. The bit-identity column, unlike the clock, does not
vary.

Bit identity is the point, not a bonus. A faster voice engine that does not produce the same
waveform is a different model, so [`bench/verify.py`](bench/verify.py) checks the waveform
before it looks at the clock.

## Why it is this fast

Ming-3.0-tts-0.5B is a Qwen2.5-0.5B backbone **plus** a diffusion head, and the head is where
the time goes. The config says `patch_size: 1`, so one autoregressive step yields one latent
frame, and the audio tokenizer's `hop_size: 882` with its own `patch_size: 4` makes that frame
3528 samples at 44.1 kHz — exactly **80 ms of audio**. Per step:

| | layers | shape | times per step |
|---|---|---|---|
| Qwen2.5 backbone | 24 | 1 token | 1 |
| DiT flow head | 8 | batch 2 (guidance), 3 tokens | **10** |
| aggregator | 8 | 2 tokens | 1 |

That is ~112 layer passes for 80 ms of audio, 80 of them in the head. Real time therefore
needs 12.5 steps per second, or 80 ms per step. Upstream takes 229 ms, so the gap is 3x, not
30x — and none of it is arithmetic.

Those tensors are tiny. What gets paid is the **dispatch** of a few thousand kernels per step,
and this machine takes 13-18 microseconds to launch a trivial kernel where Linux takes 5-10,
because Windows runs the WDDM driver model. Quantization would not touch any of that.

Measured per component by [`bench/profile_step.py`](bench/profile_step.py), with and without
the graft, on the same prompt:

| component | upstream | faster-ming |
|---|---|---|
| flow head, 10 steps | 167.06 ms (75%) | 19.77 ms (66%) |
| backbone, 1 token | 41.06 ms (18%) | 7.88 ms (26%) |
| aggregator | 15.76 ms (7%) | 1.91 ms (6%) |
| stop head + host sync | 0.26 ms | 0.24 ms |
| **one step** | **224.14 ms — 0.36x real time** | **29.80 ms — 2.68x real time** |

The flow head is still two thirds of what is left, and it is now within about 3x of what its
weight traffic alone would cost, so the remaining overhead is intra-graph kernel latency on
tiny tensors. Going further means fusing kernels, not scheduling them better.

### The three changes

**1. The flow-matching head in one CUDA graph.** The loop of `CFM.sample` is captured *as it
stands*, calling the real `forward_with_cfg`, so nothing in the model is rewritten and nothing
can drift. Two things stay outside the capture: `get_epss_timesteps`, which builds a tensor
from a Python list (a pageable host-to-device copy is not capturable), and both noise draws,
which stay in the global RNG stream so a fixed seed still yields the same voice.

**2. The vocabulary head muted.** Upstream's sampler reads `hidden_states[-1]` and never looks
at the logits, yet `lm_head` is 896 x 151936 — 272 MB of weights re-read every step, the
largest tensor in the model, computed and discarded. Returning a one-channel view costs no
kernel at all. `undo()` restores the real head, which `sample_text` does need.

**3. The decode step graphed on a static cache.** A graph needs shapes that never move, so the
growing dynamic cache has to go. Prefill still runs on the original path — it happens once per
utterance, so speeding it up buys nothing, and leaving it untouched removes a source of
divergence for free — and its keys and values are transplanted into the static cache.

A static cache shows attention 512 positions and masks the rest to negative infinity, where a
dynamic cache shows exactly the N filled ones. I expected that to cost bit identity, because
the kernel tiles 512 keys instead of N and the accumulation order could change. **It does not.**
Adding exact zeros to a sum never moves it, in any order. The prediction was wrong and the
measurement is what counts.

## The one knob that is not bit-identical, and why it stays off

The flow head integrates in 10 steps by default, and after the graft it is still two thirds of
a decode step. Upstream's sampler ships pruned timestep tables for 5, 6, 7, 10, 12 and 16
steps (`get_epss_timesteps`), so fewer steps is a supported mode, not a hack:

```python
accelerate(tts.model, steps=5)      # 4.2x real time instead of 2.8x — but listen first
```

| steps | real time | vs 10 steps | waveform | verdict |
|---|---|---|---|---|
| 10 | 2.78x | baseline | identical to upstream, to the bit | **keep** |
| 7 | 3.46x | +24% | different | rejected by ear |
| 5 | 4.19x | +51% | different | rejected by ear |

Whether "different" means "worse" is not something a number can answer here. An embedding
distance calls a hundredth of a second of shift "different"; a transcription would not notice
a mangled word at all. [`bench/flow_steps.py`](bench/flow_steps.py) therefore scores nothing:
it renders the same sentences at each step count and writes the files out for listening.

**A listener kept 10 and rejected both 7 and 5.** Worth noting that 7 was rejected too: the
quality does not taper off, it falls off a cliff somewhere between 10 and 7. That is a reason
to be suspicious of pruned-schedule speedups in general, not only here. So `steps` stays at 10
by default, and the knob is documented rather than recommended.

If you do evaluate it yourself, start with the shortest sentence. On two seconds and four words
the prosody has no room to dilute, so a weak integration shows up immediately — and listen to
consonant attacks and word endings, which is where too few steps smears first.

## The trap, for anyone capturing their own graphs

A CUDA graph records **addresses, not values**. Every tensor it reads must stay alive for as
long as the graph does, *and that includes the constants*.

The first version of this package kept the flow sampler's timestep tensor in a local variable
of the build function. When that function returned, the tensor was collected, its device memory
went back to the caching allocator, and the next large allocation wrote over it. The graph then
replayed with arbitrary timesteps.

The symptom is deceptive in the worst way: **exact right after capture, wrong later**. In
isolation every test passed. In the real decode loop the backbone allocated between replays and
the output diverged at the very first step, by a magnitude that said "wrong input", not
"rounding". Two benches chased the wrong suspects first — the cuBLAS workspace, then
x_transformers' rotary frequency cache — and both cleared CUDA of any wrongdoing.

That is why [`faster_ming/capture.py`](faster_ming/capture.py) has a `Captured` dataclass whose
only job is to hold the graph *and* everything it touches, and why nothing a graph reads is ever
left in a local.

## Run the benches yourself

```bash
python bench/verify.py \
    --ming /path/to/Ming-3.0-tts \
    --weights /path/to/Ming-3.0-tts-0.5B \
    --reference /path/to/reference.wav

python bench/profile_step.py --ming ... --weights ... --reference ...
python bench/profile_step.py --ming ... --weights ... --reference ... --accelerate
```

`profile_step.py` starts by timing a trivial kernel launch, because without that floor "a step
costs 229 ms" does not tell you whether you are compute-bound or dispatch-bound — and that is
the whole difference between reaching for quantization and reaching for CUDA graphs.

Tests:

```bash
FASTER_MING_REPO=... FASTER_MING_WEIGHTS=... FASTER_MING_REFERENCE=... pytest
```

They skip without those variables, so the suite is runnable by someone reading the code who has
no 3 GB checkpoint on disk.

## What this does not do

- **It does not change the voice.** Same weights, same output, same quality, same emotional
  range. If you want a different voice you still need a different model.
- **It does not stream.** Upstream generates every latent frame and then decodes the VAE once,
  deliberately: a chunked decode produces slightly different audio at the chunk boundaries.
  Throughput is 2.8x real time, but time-to-first-sample is still the whole utterance.
- **Batch size 1 only**, which is what upstream's sampler assumes anyway.
- **One utterance at a time per grafted model.** The backbone owns a single static cache.
- **The numbers are from one laptop.** The win comes from removing dispatch overhead, so it
  will be largest where dispatch is most expensive: Windows/WDDM, laptop GPUs, small batches.
  On Linux with a datacenter card, expect less. Run the bench.

## Requirements

- the upstream inference repo on the import path — the Hugging Face checkpoint alone is not
  enough
- `torch` with CUDA (developed on 2.6.0+cu124), `transformers >= 4.46` for `StaticCache` with
  `cache_position`
- upstream pins `torch==2.6.0` and `transformers==4.46.3`, which wants its own environment

## License

MIT, matching upstream.
