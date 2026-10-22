"""The waveform must not move. That is the whole contract.

These tests need a GPU, the upstream inference repo, the checkpoint and a
reference wav, so they are skipped unless three environment variables point at
them:

    FASTER_MING_REPO        the inclusionAI/Ming-3.0-tts inference repo
    FASTER_MING_WEIGHTS     the Ming-3.0-tts-0.5B checkpoint
    FASTER_MING_REFERENCE   a wav file to clone from

Skipping rather than failing is deliberate: the test suite should be runnable
by someone reading the code who has no 3 GB checkpoint on disk, and a test that
cannot run is not a test that failed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.environ.get("FASTER_MING_REPO")
WEIGHTS = os.environ.get("FASTER_MING_WEIGHTS")
REFERENCE = os.environ.get("FASTER_MING_REFERENCE")
SEED = 123

needs_model = pytest.mark.skipif(
    not (REPO and WEIGHTS and REFERENCE),
    reason="set FASTER_MING_REPO, FASTER_MING_WEIGHTS and "
           "FASTER_MING_REFERENCE to run the model tests",
)


@pytest.fixture(scope="module")
def tts():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    from loguru import logger
    logger.remove()
    import tqdm
    real = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = lambda self, *a, **k: real(
        self, *a, **{**k, "disable": True})

    from ming_tts import MingTTS
    # The prompt cache would make the first render consume the VAE posterior's
    # draws and the second not, so the two paths would start from offset seeds.
    return MingTTS(WEIGHTS, device="cuda:0", cache_prompts=False)


def render(tts, **kw):
    import torch
    torch.manual_seed(SEED)
    wav = tts.synthesize(prompt_wav_path=REFERENCE, use_spk_emb=True, **kw)
    torch.cuda.synchronize()
    return wav.float().reshape(-1).cpu().numpy()


@needs_model
@pytest.mark.parametrize("case", [
    pytest.param(dict(task="TTS", text="Non. Pas comme ca.", language_id="fr",
                      raw_text=True), id="short"),
    pytest.param(dict(task="TTS", text="I don't know, maybe it's simpler "
                                       "than that.", language_id="en",
                      raw_text=True), id="english"),
    pytest.param(dict(task="EMOTION", text="La page n'est pas a jour.",
                      language_id="fr",
                      controls={"emotion": "愤怒"}, cfg=3.0,
                      raw_text=True), id="emotion-cfg3"),
])
def test_waveform_is_bit_identical(tts, case):
    """Grafted output must equal upstream output, sample for sample.

    Not "close enough": none of the three stages touches the computation, so
    any difference at all is a bug, not a tradeoff.
    """
    import faster_ming

    upstream = render(tts, **case)
    graft = faster_ming.accelerate(tts.model)
    try:
        assert not graft.failures, graft.failures
        # A throwaway render so the captures do not land inside the measured
        # one and shift the random stream after them.
        render(tts, **case)
        grafted = render(tts, **case)
    finally:
        graft.undo()

    assert grafted.size == upstream.size
    assert np.array_equal(upstream, grafted), (
        f"max delta {np.abs(upstream - grafted).max():.3e}")


@needs_model
def test_undo_restores_upstream(tts):
    """After undo() the model must behave exactly as if nothing happened."""
    import faster_ming

    case = dict(task="TTS", text="Non. Pas comme ca.", language_id="fr",
                raw_text=True)
    upstream = render(tts, **case)
    faster_ming.accelerate(tts.model).undo()
    assert np.array_equal(upstream, render(tts, **case))


@needs_model
def test_split_text_path(tts):
    """The multi-fragment path re-enters generate and must still match.

    This is the only path where static-cache state could wrongly survive from
    one fragment to the next, so it gets its own test.
    """
    import faster_ming

    long_text = (
        "Il y a trois boites dans le couloir, et la fenetre du salon est "
        "restee ouverte toute la nuit, ce qui explique pourquoi le parquet "
        "est humide ce matin."
    )
    case = dict(task="TTS", text=long_text, language_id="fr", raw_text=False)
    upstream = render(tts, **case)
    graft = faster_ming.accelerate(tts.model)
    try:
        render(tts, **case)
        grafted = render(tts, **case)
    finally:
        graft.undo()
    assert np.array_equal(upstream, grafted)


def test_captured_holds_everything():
    """A graph reads addresses, so `Captured` must keep its tensors alive.

    Runs without a GPU: it checks the dataclass contract, which is the one
    thing that made the first version of this package silently wrong.
    """
    from faster_ming.capture import Captured

    fields = Captured.__dataclass_fields__
    assert "held" in fields, "constants a graph reads must be held somewhere"
    assert "inputs" in fields and "output" in fields
