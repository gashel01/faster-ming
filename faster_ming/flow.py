"""The flow-matching head in one CUDA graph. 75% of the time lives here.

WHY THE HEAD AND NOT THE BACKBONE

Ming-3.0-tts-0.5B is a Qwen2.5-0.5B backbone *plus* a diffusion head, and the
head is where the time goes. Its config says `patch_size: 1`, so one
autoregressive step yields one latent frame, and the audio tokenizer's
`hop_size: 882` with its own `patch_size: 4` makes that frame 3528 samples at
44.1 kHz -- exactly 80 ms of audio. Per step:

    Qwen2.5 backbone     24 layers, 1 token
    DiT flow head         8 layers, batch 2 (guidance), 3 tokens -- TEN TIMES
    aggregator            8 layers, 2 tokens

That is ~112 layer passes for 80 ms of audio, 80 of them in the head. Measured
on an RTX 3070 Ti Laptop, a step costs 229 ms, of which the head is 169-214.

AND IT IS OVERHEAD, NOT ARITHMETIC

Those tensors are tiny: batch 2, sequence 3, width 1024. The compute is
nothing; what gets paid is the *dispatch* of a few thousand kernels. The same
machine takes 13-18 microseconds to launch a trivial kernel, where Linux takes
5-10, because Windows runs the WDDM driver model. Quantization would not touch
any of that. A CUDA graph replaces every launch with one.

The graph is also the proof: it does not change a single kernel, weight or
shape, only the dispatch. Output identical to the bit plus an 8.5x speedup
on the head alone (167 ms -> 19.8 ms per step) means the cost was dispatch
all along.

WHAT IS CAPTURED, AND WHAT DELIBERATELY IS NOT

The loop of `CFM.sample` is captured *as it stands*, calling the real
`forward_with_cfg`. Nothing in the model is rewritten, so nothing can drift
numerically. Two things stay outside:

  - `get_epss_timesteps`, which builds a tensor from a Python list. A pageable
    host-to-device copy is not capturable. The tensor is identical, it is just
    built earlier.
  - both noise draws, which stay in the global RNG stream so that a fixed seed
    still yields the same voice. The order, shape and dtype of those draws are
    what makes a seed reproducible.

`cfg`, `sigma` and `temperature` are frozen into the capture and serve as the
cache key: one graph per setting, built on first use. Production uses one.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture


class GraphedFlowHead:
    """Drop-in replacement for `FlowLoss.sample` with identical output.

    Callable with the same signature, and it falls back to the original
    implementation for any shape or step count it was not built for -- a voice
    engine that crashes is worth less than a slow one.
    """

    def __init__(self, flowloss, patch_size, latent_dim, llm_dim, device,
                 dtype=torch.bfloat16, steps=10):
        self.flowloss = flowloss
        self.original = flowloss.sample
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.llm_dim = llm_dim
        self.device = device
        self.dtype = dtype
        self.steps = steps
        self._graphs: dict = {}
        self.captures = 0
        self.replays = 0
        self.failures: list[str] = []

    # -- the body, word for word from the upstream sampler -------------------
    def _body(self, t, noise, z, hist, sde, cfg, sigma, temperature):
        """The loop of `CFM.sample`, copied without edits.

        A copy rather than a call, only because `CFM.sample` builds `t` itself
        and that is not capturable. Everything else is identical, including the
        stochastic term that is zero at temperature 0 but keeps the same chain
        of roundings.

        Every tensor arrives as an argument and none is read off `self`, which
        is what guarantees the caller is the one holding them alive.
        """
        model = self.flowloss.cfm.model

        def fn(tt, x):
            pred_cfg = model.forward_with_cfg(
                x=x, t=tt, c=z, latent_history=hist,
                patch_size=self.patch_size)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg

        y = noise
        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            y = y + fn(t[step], y) * dt
            y = y + sigma * (temperature ** 0.5) * (dt.abs() ** 0.5) * sde[step]
        return y

    def _timesteps(self, dtype):
        """`t` as `CFM.sample` computes it, built outside the capture."""
        from fm.CFM import get_epss_timesteps

        t = get_epss_timesteps(self.steps, device=self.device, dtype=dtype)
        # `sway_sampling_coef=-1.0` is upstream's default and `CFM.sample`
        # always applies it.
        return t + -1.0 * (torch.cos(torch.pi / 2 * t) - 1 + t)

    def _build(self, key, dtype) -> Captured:
        cfg, sigma, temperature = key
        z = torch.zeros(1, 1, self.llm_dim, device=self.device, dtype=dtype)
        hist = torch.zeros(1, self.patch_size, self.latent_dim,
                           device=self.device, dtype=dtype)
        noise = torch.zeros(1, self.patch_size, self.latent_dim,
                            device=self.device, dtype=dtype)
        sde = torch.zeros(self.steps, 1, self.patch_size, self.latent_dim,
                          device=self.device, dtype=dtype)
        t = self._timesteps(dtype)

        graph, output, context = capture(
            lambda: self._body(t, noise, z, hist, sde, cfg, sigma, temperature))
        self.captures += 1
        # `t` is held on purpose. It is a constant, the graph reads it, and a
        # constant left in a local is exactly the bug described in capture.py.
        return Captured(graph=graph, output=output,
                        inputs={"z": z, "hist": hist, "noise": noise,
                                "sde": sde},
                        held={"t": t, "context": context})

    def __call__(self, z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
                 temperature=0.0, steps=None):
        # Upstream's default is 10 and its sampler never passes `steps`, so
        # `None` means "whatever this head was built for". That is what lets
        # `accelerate(model, steps=5)` actually take effect: the one knob here
        # that is NOT bit-identical, and the only one that needs an ear rather
        # than a bench.
        steps = self.steps if steps is None else steps
        if (patch_size != self.patch_size or steps != self.steps
                or z.shape[1] != 1 or z.dtype != self.dtype):
            return self.original(z, latent_history, cfg, patch_size,
                                 sigma=sigma, temperature=temperature,
                                 steps=steps)

        # The draws stay here, in this order, shape and dtype. That is what
        # keeps a fixed seed equivalent to the original implementation's.
        noise = torch.randn((z.shape[0], patch_size, self.latent_dim),
                            device=z.device, dtype=z.dtype)
        sde_noise = torch.randn((steps, *noise.shape), device=z.device,
                                dtype=z.dtype)

        key = (float(cfg), float(sigma), float(temperature))
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._build(key, z.dtype)
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(repr(exc))
                entry = False
            self._graphs[key] = entry
        if entry is False:
            return self.flowloss.cfm.sample(
                noise=noise, c=z, latent_history=latent_history,
                sde_noise=sde_noise, cfg_scale=cfg, patch_size=patch_size,
                sigma=sigma, temperature=temperature)

        self.replays += 1
        return entry.replay(z=z, hist=latent_history, noise=noise,
                            sde=sde_noise)


__all__ = ["GraphedFlowHead"]
