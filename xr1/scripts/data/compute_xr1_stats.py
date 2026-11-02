# Normalization stats + hydra data config for a SONIC dataset (added for LocoManiLab integration).
"""XR-1 keeps its normalization constants *inside* the hydra data config rather than in
a stats file, so "computing stats" and "writing the data config" are one step.

    mean / std : (action_length, 80), per position in the chunk, over the used action
                 dims only. Reserved dims stay 0/0, which keeps
                 `normalize_action_parts_sonic` from touching them.
    q01 / q99  : (1, 60) quantiles of the 43-D proprio vector, zero-padded to 60.

The chunk windows mirror `JsonDataset` exactly: every frame is a window start, and a
window that runs past the end repeats its last row (`JsonDataset._pad`). Our actions
are absolute, so the 30 rows come out nearly identical -- upstream's grow with the
horizon because theirs are deltas.

Constant dims are forced to q01 = q99 = 0. `io.validate_quantiles` raises on
`q99 <= q01` unless *both* are zero, and a joint that never moves in one task
(common in SONIC) would otherwise abort training at dataloader construction.

    python scripts/data/compute_xr1_stats.py \
        --json-dir /path/to/xr1_data/<ds> \
        --name locomanilab_<ds> --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mibot.utils.sonic import ACTION_DIM_SONIC, ACTION_PARTS_SONIC, PROPRIO_DIM_SONIC  # noqa: E402
from mibot.utils.io import STATE_DIM  # noqa: E402

ACTION_FIELDS = ("motion_token", "left_hand_joints", "right_hand_joints")


def _windows(values: np.ndarray, action_length: int) -> np.ndarray:
    """(N, D) -> (N, action_length, D), last row repeated past the episode end."""
    count = len(values)
    index = np.minimum(np.arange(count)[:, None] + np.arange(action_length)[None, :], count - 1)
    return values[index]


def accumulate(json_dir: Path, action_length: int):
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON episodes under {json_dir}")

    total = np.zeros((action_length, ACTION_DIM_SONIC), dtype=np.float64)
    total_sq = np.zeros((action_length, ACTION_DIM_SONIC), dtype=np.float64)
    count = 0
    states = []

    for path in files:
        traj = json.loads(path.read_text())
        parts = [np.asarray(traj["actions"][field], dtype=np.float32) for field in ACTION_FIELDS]
        packed = np.zeros((len(parts[0]), ACTION_DIM_SONIC), dtype=np.float32)
        for (_, slc), values in zip(ACTION_PARTS_SONIC, parts):
            packed[:, slc] = values

        chunks = _windows(packed, action_length).astype(np.float64)
        total += chunks.sum(axis=0)
        total_sq += (chunks**2).sum(axis=0)
        count += len(chunks)
        states.append(np.asarray(traj["proprios"]["joint_pos"], dtype=np.float32))

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(variance)

    used = np.zeros(ACTION_DIM_SONIC, dtype=bool)
    for _, slc in ACTION_PARTS_SONIC:
        used[slc] = True
    mean[:, ~used] = 0.0
    std[:, ~used] = 0.0

    state = np.concatenate(states, axis=0)
    q01 = np.zeros((1, STATE_DIM), dtype=np.float64)
    q99 = np.zeros((1, STATE_DIM), dtype=np.float64)
    q01[0, :PROPRIO_DIM_SONIC] = np.percentile(state, 1, axis=0)
    q99[0, :PROPRIO_DIM_SONIC] = np.percentile(state, 99, axis=0)

    degenerate = q99 <= q01
    if degenerate.any():
        frozen = [int(i) for i in np.where(degenerate[0, :PROPRIO_DIM_SONIC])[0]]
        if frozen:
            print(f"note: proprio dims {frozen} never vary; zeroing their quantiles so they are skipped")
    q01[degenerate] = 0.0
    q99[degenerate] = 0.0

    return mean, std, q01, q99, count, len(files)


def _rows(matrix: np.ndarray, indent: str, decimals: int) -> str:
    return "\n".join(
        f"{indent}- [" + ", ".join(f"{value:.{decimals}f}" for value in row) + "]" for row in matrix
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-dir", required=True, help="output of locomanilab_sonic_to_xr1_json.py")
    parser.add_argument("--name", default=None, help="config stem (default: locomanilab_<sanitized json-dir name>)")
    parser.add_argument("--config-dir", default=None,
                        help="where to write <name>.yaml (default: <repo>/configs/data)")
    parser.add_argument("--action-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="per-device batch size")
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve() if args.config_dir else (
        Path(__file__).resolve().parents[2] / "configs" / "data"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", json_dir.name).strip("_")
    name = args.name if args.name is not None else f"locomanilab_{stem}"
    target = config_dir / f"{name}.yaml"

    mean, std, q01, q99, windows, episodes = accumulate(json_dir, args.action_length)

    target.write_text(
        "# @package _global_\n"
        f"# Generated by scripts/data/compute_xr1_stats.py from {json_dir}\n"
        f"# {episodes} episodes / {windows} chunk windows / action_length {args.action_length}\n"
        "\n"
        "data:\n"
        "  type: SonicDataModule\n"
        "  params:\n"
        "    type: sonic\n"
        "    max_steps: ${trainer.max_steps}\n"
        "    train_datasets:\n"
        f"      batch_size: {args.batch_size}\n"
        f"      action_length: {args.action_length}\n"
        "      paths:\n"
        f"      - {json_dir}\n"
        "      mean:\n" + _rows(mean, "      ", args.decimals) + "\n"
        "      std:\n" + _rows(std, "      ", args.decimals) + "\n"
        "      q01:\n" + _rows(q01, "      ", args.decimals) + "\n"
        "      q99:\n" + _rows(q99, "      ", args.decimals) + "\n"
    )
    print(f"wrote {target}")
    print(f"  mean {mean.shape} std {std.shape} q01/q99 {q01.shape}")
    print(f"  action std (chunk mean, used dims): {std[:, :78].mean():.6f}")


if __name__ == "__main__":
    main()
