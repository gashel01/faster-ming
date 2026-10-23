# Samples

Rendered by `bench/samples.py` on NVIDIA GeForce RTX 3070 Ti Laptop GPU, torch 2.6.0+cu124, faster-ming 0.1.0, seed 123. Reference voice: `data/wavs/00000309-00000300.wav` from the upstream inference repo.

Each file is the faster-ming render. The upstream render of the same sentence was compared to it sample by sample before it was written; both hashes are below, over the float32 samples before 16-bit conversion.

| file | audio | upstream | faster-ming | speedup | identical | SHA-256 upstream | SHA-256 faster-ming |
|---|---|---|---|---|---|---|---|
| [fr.wav](fr.wav) | 5.00 s | 12.11 s | 1.84 s | 6.58x | yes | `842160844aea4eb6` | `842160844aea4eb6` |
| [en.wav](en.wav) | 4.60 s | 10.50 s | 1.78 s | 5.91x | yes | `ac034b0e7487e58f` | `ac034b0e7487e58f` |
| [zh.wav](zh.wav) | 4.00 s | 9.39 s | 1.60 s | 5.86x | yes | `78f408a7abc64414` | `78f408a7abc64414` |
| [en-anger.wav](en-anger.wav) | 3.80 s | 9.64 s | 1.66 s | 5.82x | yes | `fcc50db166d6142b` | `fcc50db166d6142b` |
| [zh-anger.wav](zh-anger.wav) | 4.40 s | 15.27 s | 1.73 s | 8.83x | yes | `5a784c3f93fe5b69` | `5a784c3f93fe5b69` |
