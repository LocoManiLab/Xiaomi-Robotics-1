# Widen the released checkpoint's action space (added for LocoManiLab integration).
"""Re-dim the three action-facing projectors of a released XR-1 checkpoint.

The released `model_states.pt` is built for `action_shape = (30, 60)`. LocoManiLab's G1
SONIC action space needs 80 dims (see `mibot/utils/sonic.py`), and
`BaseRunner.configure_model` loads the checkpoint with `strict=False` but *raises*
on any missing/unexpected key -- and a shape mismatch raises even earlier. So the
widening has to happen offline, on the checkpoint, before training starts.

Only three tensors touch the action dimension (`XR1._build_model`):

    action_projector.layers.0.weight          [1024, D]        input  side
    action_output_layer.layers.2.weight       [D, 1024]        output side
    action_projector_choice.1.layers.0.weight [n_choices*D, H] output side, choice-major

They are zero-padded along the action axis: the first 60 dims keep the pretrained
mapping bit-for-bit, the new dims start at exactly 0 and learn from there. This is
the standard vocabulary-resize move -- strictly better than re-initialising the
whole head, and fully reversible (the new slice is provably all-zero).

`action_projector_choice` is choice-major: `compute_choice_loss` reshapes its output
as `(length, n_choices, action_dim)`, so the weight rows are
`[choice0 dim0..dimD-1, choice1 dim0..dimD-1, ...]` and must be padded per choice
block, not appended at the end.

State is NOT touched: 43 <= 60, so the G1 proprio vector rides in the existing
padding and `state_projector` / `state_projector_choice` load unchanged.

Usage:
    python tools/widen_action_dim.py \
        --input  pretrained_ckpt/model_states.pt \
        --output pretrained_ckpt/model_states_sonic80.pt \
        --action-dim 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

PREFIX = "model."

#: key suffix -> which axis of the tensor carries the action dimension
INPUT_SIDE = "action_projector.layers.0.weight"
OUTPUT_SIDE = "action_output_layer.layers.2.weight"
CHOICE_SIDE = "action_projector_choice.1.layers.0.weight"


def _pad_rows(tensor: torch.Tensor, new: int) -> torch.Tensor:
    out = tensor.new_zeros((new, *tensor.shape[1:]))
    out[: tensor.shape[0]] = tensor
    return out


def _pad_cols(tensor: torch.Tensor, new: int) -> torch.Tensor:
    out = tensor.new_zeros((tensor.shape[0], new))
    out[:, : tensor.shape[1]] = tensor
    return out


def _pad_choice(tensor: torch.Tensor, old_dim: int, new_dim: int, n_choices: int) -> torch.Tensor:
    if tensor.shape[0] != n_choices * old_dim:
        raise ValueError(
            f"{CHOICE_SIDE} has {tensor.shape[0]} rows, expected n_choices*action_dim "
            f"= {n_choices}*{old_dim} = {n_choices * old_dim}"
        )
    blocks = tensor.view(n_choices, old_dim, tensor.shape[1])
    out = tensor.new_zeros((n_choices, new_dim, tensor.shape[1]))
    out[:, :old_dim] = blocks
    return out.reshape(n_choices * new_dim, tensor.shape[1])


def widen(state: dict, new_dim: int, n_choices: int) -> tuple[dict, int]:
    def key(suffix: str) -> str:
        for candidate in (PREFIX + suffix, suffix):
            if candidate in state:
                return candidate
        raise KeyError(f"checkpoint has no '{suffix}' (keys look like: {sorted(state)[:3]})")

    k_in, k_out, k_choice = key(INPUT_SIDE), key(OUTPUT_SIDE), key(CHOICE_SIDE)
    old_dim = state[k_in].shape[1]
    if old_dim == new_dim:
        raise SystemExit(f"checkpoint already has action_dim={new_dim}; nothing to do")
    if old_dim > new_dim:
        raise SystemExit(f"refusing to shrink action_dim {old_dim} -> {new_dim}")
    if state[k_out].shape[0] != old_dim:
        raise ValueError(f"{k_out} first dim is {state[k_out].shape[0]}, expected {old_dim}")

    out = dict(state)
    out[k_in] = _pad_cols(state[k_in], new_dim)
    out[k_out] = _pad_rows(state[k_out], new_dim)
    out[k_choice] = _pad_choice(state[k_choice], old_dim, new_dim, n_choices)
    return out, old_dim


def verify(before: dict, after: dict, old_dim: int, new_dim: int, n_choices: int) -> None:
    """Every untouched key bit-identical; every new slice exactly zero."""
    touched = {k for k in after if after[k].shape != before[k].shape}
    if len(touched) != 3:
        raise AssertionError(f"expected exactly 3 reshaped tensors, got {len(touched)}: {sorted(touched)}")
    for name in before:
        if name in touched:
            continue
        if not torch.equal(before[name], after[name]):
            raise AssertionError(f"untouched tensor changed: {name}")

    for name in touched:
        old, new = before[name], after[name]
        if name.endswith(INPUT_SIDE):
            if not torch.equal(new[:, :old_dim], old) or new[:, old_dim:].abs().sum() != 0:
                raise AssertionError(f"{name}: bad column padding")
        elif name.endswith(OUTPUT_SIDE):
            if not torch.equal(new[:old_dim], old) or new[old_dim:].abs().sum() != 0:
                raise AssertionError(f"{name}: bad row padding")
        elif name.endswith(CHOICE_SIDE):
            blocks_old = old.view(n_choices, old_dim, -1)
            blocks_new = new.view(n_choices, new_dim, -1)
            if not torch.equal(blocks_new[:, :old_dim], blocks_old) or blocks_new[:, old_dim:].abs().sum() != 0:
                raise AssertionError(f"{name}: bad per-choice padding")
        else:
            raise AssertionError(f"unexpected reshaped tensor: {name}")
    print(f"verified: {len(before)} tensors, 3 widened {old_dim}->{new_dim}, rest bit-identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="pretrained_ckpt/model_states.pt")
    parser.add_argument("--output", default="pretrained_ckpt/model_states_sonic80.pt")
    parser.add_argument("--action-dim", type=int, default=80)
    parser.add_argument("--n-choices", type=int, default=5, help="must match xr1.n_choices")
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    if target.exists():
        raise SystemExit(f"{target} already exists; delete it first")

    print(f"loading {source} ...")
    ckpt = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if "module" not in ckpt:
        raise SystemExit(f"{source} has no 'module' key (found {sorted(ckpt)[:5]})")
    state = ckpt["module"]

    widened, old_dim = widen(state, args.action_dim, args.n_choices)
    verify(state, widened, old_dim, args.action_dim, args.n_choices)

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**{k: v for k, v in ckpt.items() if k != "module"}, "module": widened}, target)
    print(f"saved {target} ({target.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
