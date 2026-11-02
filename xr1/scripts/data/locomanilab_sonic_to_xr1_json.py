# LocoManiLab-prepared G1 SONIC LeRobot -> XR-1 JSON episodes (added for LocoManiLab integration).
"""XR-1 trains from one JSON per episode plus its videos (`docs/data_format.md`);
lab collects LeRobot v2.1 + `meta/modality.json`. This writes the JSON side.

Videos are NOT re-encoded or copied: LocoManiLab's datasets already store one mp4 per
episode with a 1:1 frame correspondence to the parquet rows, so each JSON just
points `observations.ego[0].path` at the source file by absolute path.

Only one camera: LocoManiLab's G1 datasets carry `observation.images.ego_view` and nothing
else, so the prompt drops upstream's Left-/Right-Wrist blocks. `JsonDataset._images`
and `_messages` are generic over the number of `<image>` placeholders, so the
single-view prompt needs no loader change.

Action keys are selected EXPLICITLY (`action.motion_token` + the two
`teleop.*_hand_joints`). A glob over `action*` would also pick up `action.wbc` (29-D)
and silently produce a 107-D action space -- the failure that cost a day on PRTS.

    python scripts/data/locomanilab_sonic_to_xr1_json.py \
        --data-root /path/to/datasets/bench_sonic_BathroomApproachToilet \
        --out-dir   /path/to/xr1_data/bench_sonic_BathroomApproachToilet
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

#: source parquet column -> (json field, expected width)
ACTION_COLUMNS = (
    ("action.motion_token", "motion_token", 64),
    ("teleop.left_hand_joints", "left_hand_joints", 7),
    ("teleop.right_hand_joints", "right_hand_joints", 7),
)
STATE_COLUMN = ("observation.state", "joint_pos", 43)
VIDEO_KEY = "observation.images.ego_view"

#: Single-view variant of upstream's prompt (`docs/data_format.md`). `JsonDataset._prompt`
#: appends " /no_cot" and fills the empty assistant turn, so neither belongs here.
PROMPT_TEMPLATE = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n"
    "Generate robot actions for the task:\n{task}"
)


def _column(table, name: str, width: int) -> np.ndarray:
    if name not in table.schema.names:
        raise SystemExit(f"parquet is missing column '{name}' (have: {table.schema.names})")
    values = np.asarray(table.column(name).to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise SystemExit(f"column '{name}' has shape {values.shape}, expected (N, {width})")
    return values


PLACEHOLDER_TASKS = {"", "test"}


def _tasks(data_root: Path) -> dict[int, str]:
    """task_index -> task string, from meta/tasks.jsonl."""
    path = data_root / "meta" / "tasks.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["task_index"])] = row["task"]
    return out


def _episode_tasks(data_root: Path) -> dict[int, str]:
    """episode_index -> its own task string, from meta/episodes.jsonl.

    Multi-task merged datasets (all31_merged: 1611 episodes over 28 tasks) carry a
    different instruction per episode. Labelling them all with tasks.jsonl[0] would
    train the policy to ignore the language entirely -- it would see one prompt for
    28 behaviours. So the per-episode label is the authority; tasks.jsonl is only the
    fallback for datasets whose episodes.jsonl predates the `tasks` field.
    """
    path = data_root / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks = row.get("tasks") or []
        # SONIC collection writes the placeholder "test" as every episode's label; the real instruction is
        # only in tasks.jsonl (via task_index). Treat placeholders as missing so the fallback applies.
        if tasks and str(tasks[0]).strip().lower() not in PLACEHOLDER_TASKS:
            out[int(row["episode_index"])] = tasks[0]
    return out


def _task_index_of(table) -> int | None:
    """The episode's task_index from the parquet (constant within an episode)."""
    if "task_index" not in table.schema.names:
        return None
    values = table.column("task_index").to_pylist()
    return int(values[0]) if values else None


def _check_modality(data_root: Path) -> None:
    path = data_root / "meta" / "modality.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- not a LocoManiLab-prepared dataset?")
    modality = json.loads(path.read_text())
    required = {
        "video": ["ego_view"],
        "action": ["motion_token", "left_hand_joints", "right_hand_joints"],
        "state": ["joint_pos"],
    }
    missing = [
        f"{group}.{key}"
        for group, keys in required.items()
        for key in keys
        if key not in (modality.get(group) or {})
    ]
    if missing:
        raise SystemExit(f"modality.json is missing keys XR-1 needs: {', '.join(missing)}")


def convert_episode(parquet: Path, video: Path, task: str, table=None) -> dict:
    table = pq.read_table(parquet) if table is None else table
    num_frames = table.num_rows
    if num_frames <= 0:
        raise SystemExit(f"{parquet} has no rows")

    actions = {field: _column(table, column, width) for column, field, width in ACTION_COLUMNS}
    state = _column(table, STATE_COLUMN[0], STATE_COLUMN[2])
    for name, values in (*actions.items(), (STATE_COLUMN[1], state)):
        if len(values) != num_frames:
            raise SystemExit(f"{parquet}: '{name}' has {len(values)} rows, expected {num_frames}")

    return {
        "trajectory_type": "success",
        "time": parquet.stem,
        "num_frames": num_frames,
        "instruction": {
            "general": [
                {
                    "images": ["observations.ego"],
                    "conversations": [
                        {"from": "human", "value": PROMPT_TEMPLATE.format(task=task)},
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        },
        "observations": {"ego": [{"path": str(video), "start": 0, "crop_bbox": None}]},
        "proprios": {STATE_COLUMN[1]: state.tolist()},
        "actions": {field: values.tolist() for field, values in actions.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="LocoManiLab-prepared LeRobot dataset dir")
    parser.add_argument("--out-dir", required=True, help="where the per-episode JSON files go")
    parser.add_argument("--task-prompt", default=None, help="override the dataset's own task string")
    parser.add_argument("--verify-frames", action="store_true",
                        help="decode each mp4 to confirm its frame count matches the parquet")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _check_modality(data_root)

    parquets = sorted(data_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no episode parquet under {data_root}/data/chunk-*/")
    tasks = _tasks(data_root)
    episode_tasks = _episode_tasks(data_root)
    if args.task_prompt is None and not tasks and not episode_tasks:
        raise SystemExit("dataset has no meta/tasks.jsonl or per-episode tasks; pass --task-prompt")

    out_dir.mkdir(parents=True, exist_ok=True)
    written, total_frames = 0, 0
    used = Counter()
    for parquet in parquets:
        chunk = parquet.parent.name
        video = data_root / "videos" / chunk / VIDEO_KEY / f"{parquet.stem}.mp4"
        if not video.exists():
            raise SystemExit(f"missing video for {parquet.name}: {video}")

        table = pq.read_table(parquet)
        episode_index = int(parquet.stem.rsplit("_", 1)[-1])
        task = args.task_prompt or episode_tasks.get(episode_index)
        if task is None:                       # fall back to the parquet's task_index
            index = _task_index_of(table)
            task = tasks.get(index) if index is not None else None
        if task is None:
            raise SystemExit(
                f"{parquet.name}: no task label (episode_index {episode_index} is not in "
                f"meta/episodes.jsonl and the parquet has no usable task_index). "
                f"Pass --task-prompt to force one.")
        used[task] += 1

        episode = convert_episode(parquet, video, task, table=table)
        if args.verify_frames:
            from decord import VideoReader

            frames = len(VideoReader(str(video)))
            if frames != episode["num_frames"]:
                raise SystemExit(
                    f"{video.name}: {frames} video frames vs {episode['num_frames']} parquet rows"
                )

        (out_dir / f"{parquet.stem}.json").write_text(json.dumps(episode))
        written += 1
        total_frames += episode["num_frames"]

    print(f"wrote {written} episodes / {total_frames} frames -> {out_dir}")
    print(f"task labels: {len(used)} distinct over {written} episodes")
    for task, count in used.most_common():
        print(f"  {count:5d}  {task}")


if __name__ == "__main__":
    main()
