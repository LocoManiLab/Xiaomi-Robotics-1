# Unitree G1 SONIC variant of JsonDataset (added for LocoManiLab integration).
"""Reads the same JSON-per-episode container as `JsonDataset`, but with LocoManiLab's
G1 SONIC fields instead of upstream's bimanual end-effector deltas:

    actions.motion_token        (N, 64)   absolute latent motion tokens
    actions.left_hand_joints    (N, 7)
    actions.right_hand_joints   (N, 7)
    proprios.joint_pos          (N, 43)

Everything else -- sample indexing, prompt handling, video decoding, the colour
augmentation, the `<a_i>` / `<score>` turns, the trajectory-type mask -- is
inherited unchanged, so this stays in lockstep with upstream.

The two overrides exist because `JsonDataset.__getitem__` hardcodes the EE-delta
semantics (`_arm_action` / `_state`) and `__init__` validates stats against the
60-wide upstream action space.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import normalize_quantile, validate_quantiles
from mibot.utils.sonic import (
    build_action_mask_sonic,
    compose_action_sonic,
    compose_state_sonic,
    normalize_action_parts_sonic,
    validate_stats_sonic,
)

#: `actions.<key>` fields packed, in `ACTION_PARTS_SONIC` order.
ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")


class SonicJsonDataset(JsonDataset):
    def __init__(self, params):
        data = params["train_datasets"]
        self.action_length = int(data.get("action_length", params.get("action_length", 30)))
        self.batch_size = int(data.get("batch_size", 16))
        self.max_samples = (
            int(params.get("max_steps", 1000))
            * self.batch_size
            * int(os.environ.get("WORLD_SIZE", 1))
        )
        self.mean, self.std = validate_stats_sonic(data["mean"], data["std"], self.action_length)
        self.q01, self.q99 = validate_quantiles(data["q01"], data["q99"])
        self.files = self._json_files(data["paths"])
        self.samples = self._samples()

    def __getitem__(self, index):
        sample = self.samples[index]
        traj = self._read_json(sample["file"])
        frame = sample["frame_index"]
        steps = min(self.action_length, int(traj["num_frames"]) - frame)
        if steps <= 0:
            raise IndexError(f"invalid sample at index={index}: frame_index={frame}")

        prompt = self._prompt(traj)
        raw = [self._pad(self._future(traj, f"actions.{key}", frame, steps), steps) for key in ACTION_KEYS]
        action = compose_action_sonic(
            *normalize_action_parts_sonic(raw, self.mean, self.std), action_length=self.action_length
        )
        state = normalize_quantile(
            compose_state_sonic(self._frame(traj, "proprios.joint_pos", frame)), self.q01, self.q99
        )
        action_mask = build_action_mask_sonic(self.action_length, self._mask(traj, steps))
        prompt["conversations"] += [
            {"from": "human", "value": "Robot state: <state>"},
            {"from": "gpt", "value": "".join(f"<a_{index}>" for index in range(steps)) + "<score>"},
        ]

        return {
            "messages": self._messages(
                prompt["conversations"],
                self._augment(self._images(traj, prompt["images"], frame)),
            ),
            "action": torch.from_numpy(np.ascontiguousarray(action)),
            "action_mask": torch.from_numpy(action_mask),
            "state": torch.from_numpy(np.ascontiguousarray(state)),
            "vlm_action_target": torch.from_numpy(np.ascontiguousarray(action[:steps])),
            "vlm_action_mask": torch.from_numpy(action_mask[:steps]),
            "vlm_action_actual_length": steps,
        }
