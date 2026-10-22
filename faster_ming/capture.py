"""CUDA graph capture, and the one rule that makes it safe.

A CUDA graph records *addresses*, not values. Every tensor a captured graph
reads must therefore stay alive for as long as the graph does, and that
includes the constants. Miss one and you get the most misleading bug in this
whole area:

    the graph is exact right after capture, and wrong later.

The first version of this package kept the flow sampler's timestep tensor in
a local variable of the build function. When that function returned, the
tensor was collected, its device memory went back to the caching allocator,
and the next large allocation wrote over it. The graph then replayed with
arbitrary timesteps. Two benches chased the wrong suspects first -- the cuBLAS
workspace, then x_transformers' rotary frequency cache -- and both cleared
CUDA of any wrongdoing.

So `Captured` holds the graph *and* everything it touches, and nothing a graph
reads is ever left in a local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import torch


@dataclass
class Captured:
    """A replayable graph together with every tensor it reads or writes.

    `inputs` are the static buffers a caller writes into before each replay,
    `output` is the static buffer the graph writes, and `held` is for anything
    else the graph reads -- constants, precomputed tables, the capture context.
    Nothing here is decoration: dropping any of it reintroduces the bug in this
    module's docstring.
    """

    graph: torch.cuda.CUDAGraph
    output: Any
    inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    held: Dict[str, Any] = field(default_factory=dict, repr=False)

    def replay(self, **values: torch.Tensor):
        """Copy `values` into the static inputs, replay, return a fresh output.

        The output is cloned because the next replay overwrites that buffer and
        callers keep their results in lists.
        """
        for name, value in values.items():
            self.inputs[name].copy_(value)
        self.graph.replay()
        return self.output.clone()


def capture(body: Callable[[], Any], warmups: int = 3) -> tuple:
    """Warm up on a side stream, then capture `body`. Returns (graph, output).

    The side-stream warmup is required by CUDA, not optional hygiene: it is
    what forces cuBLAS to allocate its workspaces and pick its algorithms,
    neither of which can happen during a capture.

    The capture context is returned to the caller so it can be held; see this
    module's docstring for why nothing is left to go out of scope.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            body()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    context = torch.cuda.graph(graph)
    with context:
        output = body()
    return graph, output, context


class GraphedCall:
    """A graph for a callable with one fixed input shape, eager otherwise.

    Ming's aggregator is called twice per utterance with different shapes: once
    over the whole audio prompt, and once per decode step over a single latent
    frame. Only the second is hot, and only the second has a constant shape, so
    an unexpected shape falls back to the original call rather than failing.

    It takes the *bound call*, not the module: `linear_proj_audio` is a
    registered submodule and `nn.Module` refuses to have anything but a module
    assigned over it, so the graft replaces its `forward`. Holding the original
    call is also what keeps the fallback from recursing into itself.
    """

    def __init__(self, call, shape, device, dtype=torch.bfloat16):
        self.call = call
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.captured = None
        self.failures: list[str] = []
        self.replays = 0

    def _build(self) -> Captured:
        held = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        graph, output, context = capture(lambda: self.call(held))
        return Captured(graph=graph, output=output, inputs={"x": held},
                        held={"context": context})

    def __call__(self, x):
        if tuple(x.shape) != self.shape or x.dtype != self.dtype:
            return self.call(x)
        if self.captured is None:
            try:
                self.captured = self._build()
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(repr(exc))
                self.captured = False
        if self.captured is False:
            return self.call(x)
        self.replays += 1
        return self.captured.replay(x=x)


__all__ = ["Captured", "capture", "GraphedCall"]
