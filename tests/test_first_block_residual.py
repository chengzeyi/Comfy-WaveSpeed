"""The first-block residual has to be measured against a copy, not against the input object.

FBCache decides everything from

    first_residual = (output of block 0) - (input to block 0)

which needs the input as it was *before* block 0 ran. ComfyUI's transformer blocks update their
input in place and return the very object they were handed -- `img += ...` in
comfy/ldm/flux/layers.py -- so without a copy that subtraction is `x - x` and the residual is
exactly zero, at every step, for every model.

Zero is the worst value it can take, because nothing raises. `are_two_tensors_similar` computes
`(t1 - t2).abs().mean() / t1.abs().mean()`, which for two zero tensors is `0 / 0` -> nan, and
`nan < threshold` is False. So the cache reports a miss on every step forever: the node is
switched on, the output is bit-identical to the unpatched model, and the run is slightly slower
than not using it at all.

The copy used to be gated on `clone_original_hidden_states=... == "LTXVModel"`, so LTX-2.3 was
the one model it worked for.

Two tests, both on CPU with no weights:

  * `test_comfyui_blocks_update_in_place` establishes the premise against the real ComfyUI class,
    so this is not a claim about ComfyUI but a measurement of it.
  * `test_first_block_residual_is_not_zero` drives `CachedTransformerBlocks` with a block that has
    the same semantics and asserts the buffer the cache decides on is nonzero.

Run:

    python -m pytest tests/test_first_block_residual.py
    python tests/test_first_block_residual.py        # same tests, no pytest needed

ComfyUI is found by walking up from this file, which works when the node is installed under
`custom_nodes/`. Set COMFYUI_PATH to point elsewhere.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent


def _find_comfyui() -> Path | None:
    env = os.environ.get("COMFYUI_PATH")
    if env and (Path(env) / "comfy" / "model_management.py").is_file():
        return Path(env)
    for candidate in NODE_DIR.parents:
        if (candidate / "comfy" / "model_management.py").is_file():
            return candidate
    return None


COMFYUI = _find_comfyui()
if COMFYUI is None:
    raise SystemExit(
        "could not find a ComfyUI checkout above " + str(NODE_DIR) +
        "; set COMFYUI_PATH. These tests run against the real ComfyUI on purpose -- the bug "
        "they cover is a bug about disagreeing with it, and a mocked block would agree with "
        "whatever this node does.")
sys.path.insert(0, str(COMFYUI))

import torch  # noqa: E402


def load_first_block_cache():
    spec = importlib.util.spec_from_file_location(
        "wavespeed_first_block_cache_undertest", NODE_DIR / "first_block_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comfyui_blocks_update_in_place():
    """The premise, measured rather than asserted: a real DoubleStreamBlock mutates the tensor
    it is given and returns that same object."""
    import comfy.ops
    from comfy.ldm.flux.layers import DoubleStreamBlock

    torch.manual_seed(0)
    hidden, heads, img_len, txt_len = 64, 4, 16, 8
    block = DoubleStreamBlock(hidden, heads, 4.0, qkv_bias=True, dtype=torch.float32,
                              device="cpu", operations=comfy.ops.disable_weight_init).eval()
    with torch.no_grad():
        # disable_weight_init leaves parameters uninitialised. Left alone the block adds zero
        # and the test would pass for entirely the wrong reason.
        for parameter in block.parameters():
            parameter.normal_(0, 0.05)

        img = torch.randn(1, img_len, hidden)
        txt = torch.randn(1, txt_len, hidden)
        vec = torch.randn(1, hidden)
        pe = torch.randn(1, 1, img_len + txt_len, (hidden // heads) // 2, 2, 2)

        aliased = img            # what the old code kept
        before = img.clone()     # what it should have kept
        out_img, _ = block(img=img, txt=txt, vec=vec, pe=pe)

    assert out_img is aliased, "the block no longer returns its input object"
    assert not torch.equal(aliased, before), "the block no longer updates in place"
    assert (out_img - aliased).abs().max().item() == 0.0, (
        "measuring the residual against the input object gives exactly zero")
    assert (out_img - before).abs().max().item() > 0.0, (
        "measuring it against a copy gives the real residual")


class InPlaceBlock(torch.nn.Module):
    """A block with ComfyUI's semantics: updates its input and returns that same object."""

    def __init__(self, width: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.proj = torch.nn.Linear(width, width)

    def forward(self, hidden_states, encoder_hidden_states):
        hidden_states += self.proj(hidden_states)
        return hidden_states, encoder_hidden_states


def test_first_block_residual_is_not_zero():
    module = load_first_block_cache()

    width = 32
    blocks = torch.nn.ModuleList([InPlaceBlock(width, seed=i) for i in range(3)])
    cached = module.CachedTransformerBlocks(blocks, None, residual_diff_threshold=0.12)

    hidden_states = torch.randn(1, 8, width)
    encoder_hidden_states = torch.randn(1, 4, width)

    with module.cache_context(module.create_cache_context()):
        with torch.no_grad():
            cached(hidden_states, encoder_hidden_states)
        residual = module.get_buffer("first_hidden_states_residual")

    assert residual is not None, "the cache never recorded a first-block residual"
    assert residual.abs().max().item() > 0.0, (
        "the first-block residual is exactly zero, so the cache can never fire: "
        "`original_hidden_states` is aliasing the tensor block 0 updates in place")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
