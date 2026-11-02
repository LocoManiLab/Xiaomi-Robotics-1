"""Unitree G1 SONIC action/state layout for XR-1 (added for LocoManiLab integration).

Upstream XR-1 speaks one action space: `io.ACTION_PARTS`, a (30, 60) padded tensor
holding bimanual end-effector *deltas* (pos / axis-angle / gripper) plus waist and
base velocity, with dims 20-59 reserved. LocoManiLab's G1 SONIC datasets speak a different
one: 64 *absolute* latent motion tokens plus 7+7 hand joints -- 78 dims, which does
not fit upstream's 60.

So this module defines a second, parallel layout. It does NOT touch `io.py`: the
upstream constants stay valid for the upstream JsonDataset / server path, and
everything sonic-specific lives here.

    action (30, 80):  motion_token[0:64] | left_hand[64:71] | right_hand[71:78] | reserved[78:80]
    state  (1,  60):  joint_pos[0:43]                                          | reserved[43:60]

Why 80 and not 78: upstream's own design pads the action space (20 real dims in a
60-wide tensor) so a new channel never forces another weight surgery. We keep two
spare dims for the same reason. The action mask zeroes them, so they cost nothing.

Why the state is NOT widened: 43 <= `io.STATE_DIM` (60), so joint_pos rides in the
existing padding. q01/q99 are left at 0 for dims 43:60, and both the training
normalizer (`io.normalize_quantile`) and the serving one skip dims where
`q99 <= q01`. That keeps `state_projector` / `state_projector_choice` loadable from
the released checkpoint untouched -- and keeps `CustomCollate`'s
`<a_0>..<a_59>` token assertion (which is about *chunk timesteps*, not dims) valid.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mibot.utils.io import ACTION_EPS, STATE_DIM

#: Width of the packed sonic action tensor (78 used + 2 reserved).
ACTION_DIM_SONIC = 80

#: Width of the G1 proprioceptive vector that rides in the 60-wide state tensor.
PROPRIO_DIM_SONIC = 43

#: Named slices of the packed sonic action. Anything outside these is reserved and
#: masked to zero, exactly like dims 20-59 of upstream's `io.ACTION_PARTS`.
ACTION_PARTS_SONIC = (
    ("motion_token", slice(0, 64)),
    ("left_hand", slice(64, 71)),
    ("right_hand", slice(71, 78)),
)

#: The 43-D `observation.state` vector as the G1 host lays it out. Mirrors
#: `meta/modality.json` of a LocoManiLab-prepared dataset and the `state` groups the
#: benchmark client sends (`robocasa/scripts/deploy_living_room_vla.py`).
STATE_GROUPS_SONIC = (
    ("left_leg", slice(0, 6)),
    ("right_leg", slice(6, 12)),
    ("waist", slice(12, 15)),
    ("left_arm", slice(15, 22)),
    ("left_hand", slice(22, 29)),
    ("right_arm", slice(29, 36)),
    ("right_hand", slice(36, 43)),
)

#: Keys the gr00t-compat policy server returns, in packed-action order.
ACTION_OUTPUT_KEYS = (
    ("action.motion_token", slice(0, 64)),
    ("action.left_hand_joints", slice(64, 71)),
    ("action.right_hand_joints", slice(71, 78)),
)


def validate_stats_sonic(mean: Sequence[Sequence[float]], std: Sequence[Sequence[float]], action_length: int):
    """`io.validate_stats`, but against ACTION_DIM_SONIC."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    expected = (action_length, ACTION_DIM_SONIC)
    if mean.shape != expected:
        raise ValueError(f"mean expected shape {expected}, got {mean.shape}")
    if std.shape != expected:
        raise ValueError(f"std expected shape {expected}, got {std.shape}")
    return mean, std


def build_action_mask_sonic(action_length: int, temporal_mask=None) -> np.ndarray:
    """1 on the used action dims of valid timesteps, 0 elsewhere (incl. reserved dims)."""
    temporal = (
        np.ones(action_length, dtype=np.int32)
        if temporal_mask is None
        else np.asarray(temporal_mask, dtype=np.int32)
    )
    mask = np.zeros((action_length, ACTION_DIM_SONIC), dtype=np.int32)
    for _, slc in ACTION_PARTS_SONIC:
        mask[:, slc] = temporal[:, None]
    return mask


def compose_action_sonic(motion_token, left_hand, right_hand, action_length: int) -> np.ndarray:
    """Pack (T,64)+(T,7)+(T,7) into the (action_length, 80) tensor."""
    action = np.zeros((action_length, ACTION_DIM_SONIC), dtype=np.float32)
    for (_, slc), value in zip(ACTION_PARTS_SONIC, (motion_token, left_hand, right_hand)):
        action[:, slc] = np.asarray(value, dtype=np.float32)
    return action


def compose_state_sonic(joint_pos) -> np.ndarray:
    """Pack the 43-D proprio vector into the (1, 60) state tensor."""
    joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(1, -1)
    if joint_pos.shape[1] != PROPRIO_DIM_SONIC:
        raise ValueError(f"joint_pos expected {PROPRIO_DIM_SONIC} dims, got {joint_pos.shape[1]}")
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    state[:, :PROPRIO_DIM_SONIC] = joint_pos
    return state


def state_from_groups(groups: Mapping[str, np.ndarray]) -> np.ndarray:
    """Reassemble the 43-D proprio vector from the benchmark client's per-group dict.

    Each group arrives with leading [batch=1, time=1] dims; we take the last frame.
    """
    joint_pos = np.zeros(PROPRIO_DIM_SONIC, dtype=np.float32)
    for name, slc in STATE_GROUPS_SONIC:
        if name not in groups:
            raise KeyError(f"observation.state is missing group '{name}' (have {sorted(groups)})")
        value = np.asarray(groups[name], dtype=np.float32).reshape(-1)
        width = slc.stop - slc.start
        if value.size < width:
            raise ValueError(f"state group '{name}' has {value.size} values, expected {width}")
        joint_pos[slc] = value[-width:]
    return joint_pos


def normalize_state_sonic(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Quantile-normalize to [-1, 1], skipping dims where q99 <= q01 (padding/constant)."""
    state = np.asarray(state, dtype=np.float32)
    normalized = np.zeros_like(state)
    valid = q99 > q01
    normalized[valid] = 2.0 * (state[valid] - q01[valid]) / (q99[valid] - q01[valid] + ACTION_EPS) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def split_action_sonic(action) -> dict[str, np.ndarray]:
    """(T, 80) -> the three gr00t-compat action keys, float32."""
    action = np.asarray(action, dtype=np.float32)
    return {name: np.ascontiguousarray(action[..., slc]) for name, slc in ACTION_OUTPUT_KEYS}


def normalize_action_parts_sonic(values, mean, std):
    """Per-part (a - mean) / std, mirroring `io.normalize_action_parts`.

    Normalizing part-by-part (rather than the whole packed tensor) is what keeps the
    reserved dims at exactly 0 regardless of what the stats file says about them.
    """
    return tuple(
        (np.asarray(value, dtype=np.float32) - mean[:, slc]) / (std[:, slc] + ACTION_EPS)
        for (_, slc), value in zip(ACTION_PARTS_SONIC, values)
    )
