"""Drop-in PolicyServer compatible with Isaac-GR00T's ZMQ REQ/REP protocol, backed by
a post-trained Xiaomi-Robotics-1 checkpoint (SONIC / G1 78-D motion_token space).

Why this file exists
--------------------
XR-1 ships its own runtime (`mibot/server/runtime/server.py`): a raw TCP socket
speaking length-prefixed `np.savez` blobs, where the **client** owns the Qwen3-VL
processor and sends `input_ids` / `pixel_values` over the wire. LocoManiLab drives both the
real G1 and the tele_humanoid_benchmark sim through one contract instead -- a
gr00t-compatible policy server on `tcp://0.0.0.0:5550` answering `get_action` with
raw observations (ZMQ REQ/REP + msgpack-numpy). gr00t serves it natively; fastwam,
lingbot-va, starVLA, PRTS and Psi0 each wrap their model into it. This is the XR-1
equivalent, so the tokenizer/processor moves server-side.

What it does per get_action()
-----------------------------
  1. obs['video']['ego_view'] -> PIL -> resize_image(factor=32, max_pixels=160000),
     the same transform `mibot/server/runtime/client.py` applies.
  2. Chat template: the SINGLE-view variant of the client's prompt (LocoManiLab's G1
     datasets have no wrist cameras), plus " /no_cot" and the "<cot></cot>"
     assistant turn -- byte-identical to what `JsonDataset._prompt` produces at
     training time.
  3. obs['state'][left_leg|right_leg|waist|left_arm|left_hand|right_arm|right_hand]
     -> 43-D joint_pos -> (1, 60) padded -> q01/q99 normalize -> clamp[-1, 1].
  4. model.generate(batch) -> (1, 30, 80) normalized -> denormalize -> split:
        action.motion_token      [H, 64]
        action.left_hand_joints  [H, 7]
        action.right_hand_joints [H, 7]

Inference does NOT use the `<state>` / `<a_i>` turns: `xr1.forward(return_loss=False)`
takes the `else` branch, where the VLM only supplies a KV cache and state/action ride
the DiT. That is why the serving prompt is shorter than the training one.

Usage (inside the xr1 conda env, cwd = repo root):
    python deploy/run_xr1_server.py \
        --run-dir <outputs>/project_locomanilab/<exp> --host 0.0.0.0 --port 5550 \
        --initial-prompt "walk over to the toilet and stand next to it"

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REP socket on tcp://<host>:<port>
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config
Dependencies beyond `pip install -e xr1`: pyzmq, msgpack-numpy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

# `pip install -e xr1` puts mibot on the path, but running straight from a checkout
# without it should work too.
_XR1_ROOT = Path(__file__).resolve().parent.parent / "xr1"
if str(_XR1_ROOT) not in sys.path:
    sys.path.insert(0, str(_XR1_ROOT))

from mmengine import Config  # noqa: E402

from mibot.models import MIMODEL  # noqa: E402
from mibot.utils.io import ACTION_EPS, denormalize_action, resize_image  # noqa: E402
from mibot.utils.sonic import (  # noqa: E402
    build_action_mask_sonic,
    compose_state_sonic,
    split_action_sonic,
    state_from_groups,
    validate_stats_sonic,
)

VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"

#: Single-view chat template. Mirrors `runtime/client.py::_messages` with the two
#: wrist blocks removed.
def _messages(instruction: str, ego: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following observations are captured from multiple views.\n# Ego View\n",
                },
                {"type": "image", "image": ego},
                {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _find_shard(run_dir: Path, ckpt_name: str) -> Path:
    """Locate the DeepSpeed rank-0 model shard inside a Lightning checkpoint dir."""
    candidates = [
        run_dir / ckpt_name / "checkpoint" / "mp_rank_00_model_states.pt",
        run_dir / ckpt_name / "mp_rank_00_model_states.pt",
        Path(ckpt_name) / "checkpoint" / "mp_rank_00_model_states.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    available = sorted(p.name for p in run_dir.glob("*.ckpt")) if run_dir.exists() else []
    raise SystemExit(
        f"no '{ckpt_name}' shard under {run_dir} (looked for mp_rank_00_model_states.pt). "
        f"Checkpoints present: {available or 'none'}"
    )


class XR1GrootPolicy:
    def __init__(self, run_dir: Path, ckpt_name: str, device: str,
                 initial_prompt: Optional[str], exec_horizon: Optional[int],
                 prefix_len: int = 0, replan_steps: int = 20,
                 num_steps: int | None = None, chunk_gain: float = 1.0) -> None:
        self.device = device
        self._current_prompt = initial_prompt
        self.exec_horizon = exec_horizon
        self._prev_token = None
        # Async / prefix-conditioned inference. XR-1 is trained with
        # `async_train=True`: half the training samples hand the model a 1-6 step
        # ground-truth action prefix and ask it to CONTINUE. Upstream's own client
        # deploys that way (`action_prefix=` in mibot/server/runtime/client.py).
        # Serving with no prefix runs the model in the other half of its training
        # distribution, where it has no motion cue at all.
        self.prefix_len = int(prefix_len)
        self.replan_steps = int(replan_steps)
        self._prev_chunk = None      # last chunk served, DEnormalized (T, action_dim)
        # Flow-ODE integration steps. The model hardcodes 5; few-step flow sampling is
        # known to under-disperse, and this checkpoint's chunks are systematically
        # flatter than ground truth (worst at low motion: 0.4x at rest, ~0.9x while
        # walking). Raising it costs latency and nothing else.
        self.num_steps = num_steps
        # Deployment-side temporal gain: a' = mean + g * (a - mean), applied per chunk.
        # Scales how much the action VARIES across the chunk without moving its level.
        # A workaround for the under-dispersion above, not a fix for it.
        self.chunk_gain = float(chunk_gain)

        config_path = run_dir / "config.py"
        if not config_path.exists():
            raise SystemExit(f"{config_path} not found -- --run-dir must be a training run dir")
        cfg = Config.fromfile(str(config_path))

        model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)
        shard = _find_shard(run_dir, ckpt_name)
        print(f"[xr1] loading {shard}", flush=True)
        weights = torch.load(shard, map_location="cpu", mmap=True, weights_only=False)["module"]
        info = model.load_state_dict(_strip_prefix(weights, "model."), strict=True)
        print(f"[xr1] load_state_dict: {info}", flush=True)
        self.model = model.eval().to(device)
        if self.num_steps is not None:
            print(f"[xr1] flow ODE steps: {self.model.num_steps} -> {self.num_steps}", flush=True)
            self.model.num_steps = int(self.num_steps)

        data = cfg.data.params.train_datasets
        self.action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
        mean, std = validate_stats_sonic(data.mean, data.std, self.action_length)
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)
        self.q01 = np.asarray(data.q01, dtype=np.float32)
        self.q99 = np.asarray(data.q99, dtype=np.float32)
        self.action_mask = torch.from_numpy(build_action_mask_sonic(self.action_length)).to(device)

        action_dim = self.mean.shape[-1]
        model_dim = self.model.action_shape[-1]
        if action_dim != model_dim:
            raise SystemExit(
                f"stats say action_dim={action_dim} but the model was built with {model_dim}; "
                f"the run's config.py is inconsistent"
            )

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(VLM_ID)
        self.processor.tokenizer.padding_side = "right"
        print(f"[xr1] ready: action_length={self.action_length} action_dim={action_dim}", flush=True)

    # ---- public endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "xr1 gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        self._prev_token = None
        self._prev_chunk = None
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        t0 = time.monotonic()
        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description']."
            )
        self._current_prompt = prompt

        image = resize_image(Image.fromarray(self._extract_image(observation)),
                             factor=32, max_pixels=160000)
        batch = self.processor.apply_chat_template(
            [_messages(prompt, image)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        batch = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in dict(batch).items()
        }

        joint_pos = self._extract_proprio(observation)
        state = compose_state_sonic(joint_pos)                       # (1, 60), unnormalized
        normalized = np.zeros_like(state)
        valid = self.q99 > self.q01
        normalized[valid] = (
            2.0 * (state[valid] - self.q01[valid]) / (self.q99[valid] - self.q01[valid] + ACTION_EPS) - 1.0
        )
        batch["state"] = torch.from_numpy(np.clip(normalized, -1.0, 1.0))[None].to(self.device)

        mask = self.action_mask.unsqueeze(0)
        prefix = self._build_prefix(observation)
        if prefix is None:
            batch["action"] = torch.zeros((1, *self.mean.shape), device=self.device,
                                          dtype=torch.bfloat16)
        else:
            # Upstream's wire contract: the prefix arrives UNNORMALIZED and the
            # server normalizes it (mibot/server/runtime/server.py), because the
            # caller thinks in real action units, not in this checkpoint's stats.
            padded = torch.zeros((1, *self.mean.shape), device=self.device)
            padded[0, : prefix.shape[0]] = torch.from_numpy(prefix).to(self.device)
            batch["action"] = ((padded - self.mean) / (self.std + ACTION_EPS)) * mask
            batch["prefix_length"] = prefix.shape[0]
        batch["action_mask"] = mask

        t_inf = time.monotonic()
        with torch.no_grad():
            raw = self.model.generate(batch)
            raw = denormalize_action(raw * mask, self.mean, self.std) * mask
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        # bfloat16 has no numpy dtype -- .float() BEFORE .cpu().numpy(), or msgpack
        # dies with "Got unsupported ScalarType BFloat16" from inside the packer.
        chunk = raw[0].float().detach().cpu().numpy().astype(np.float32)
        if self.chunk_gain != 1.0:
            level = chunk.mean(axis=0, keepdims=True)
            chunk = level + self.chunk_gain * (chunk - level)
        if self.exec_horizon is not None:
            chunk = chunk[: self.exec_horizon]

        self._prev_chunk = chunk.copy()
        action_dict = split_action_sonic(chunk)
        # Chunk statistics, not just a heartbeat. A policy that answers every call
        # but emits a near-constant token stream drives a robot that stands still,
        # and the request count alone cannot tell that apart from a working policy.
        # `tok_std` is the within-chunk temporal std of motion_token (training data
        # sits around 0.10); `d_prev` is the jump from the previously served chunk.
        tokens = action_dict["action.motion_token"]
        tok_std = float(np.std(tokens, axis=0).mean())
        delta = ("n/a" if self._prev_token is None
                 else f"{float(np.linalg.norm(tokens[0] - self._prev_token)):.4f}")
        self._prev_token = tokens[-1].copy()
        pfx = 0 if prefix is None else prefix.shape[0]
        print(f"[xr1] get_action: H={chunk.shape[0]} infer={infer_ms:.0f} ms "
              f"tok_std={tok_std:.4f} d_prev={delta} prefix={pfx} prompt={prompt!r}", flush=True)
        info = {"inference_time_s": time.monotonic() - t0, "chunk_horizon": int(chunk.shape[0])}
        return action_dict, info

    # ---- internals ----------------------------------------------------------
    def _build_prefix(self, obs: dict) -> Optional[np.ndarray]:
        """The action prefix for this call, unnormalized (k, action_dim), or None.

        Two sources, in priority order:
          1. `obs["action_prefix"]` -- an explicit prefix from the caller, mirroring
             upstream's `action_prefix=` client argument. Lets a host that tracks what
             it actually executed supply the truth.
          2. Internal feedback (`--prefix-len K`): the previous chunk's own prediction
             for the steps this new chunk now covers. The client re-plans every
             `--replan-steps` control steps, so the new chunk's step 0 lines up with
             the previous chunk's step `replan_steps` -- that overlap is the prefix.
        """
        explicit = obs.get("action_prefix")
        if explicit is not None:
            prefix = np.asarray(explicit, dtype=np.float32)
            while prefix.ndim > 2:
                prefix = prefix[0]
            if prefix.ndim != 2:
                raise ValueError(f"action_prefix must be (k, dim), got {prefix.shape}")
            return self._fit_prefix(prefix)

        if self.prefix_len <= 0 or self._prev_chunk is None:
            return None
        start = min(self.replan_steps, len(self._prev_chunk) - 1)
        prefix = self._prev_chunk[start : start + self.prefix_len]
        return self._fit_prefix(prefix) if len(prefix) else None

    def _fit_prefix(self, prefix: np.ndarray) -> np.ndarray:
        """Clamp to the chunk length and widen a 78-wide prefix to the model's width."""
        width = self.mean.shape[-1]
        if prefix.shape[1] < width:
            padded = np.zeros((prefix.shape[0], width), dtype=np.float32)
            padded[:, : prefix.shape[1]] = prefix
            prefix = padded
        elif prefix.shape[1] > width:
            prefix = prefix[:, :width]
        return prefix[: max(1, self.action_length - 1)].astype(np.float32)

    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        raw = (obs.get("language") or {}).get("annotation.human.task_description")
        while isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        video = obs.get("video") or {}
        image = None
        for key in ("ego_view", "observation.images.ego_view"):
            if key in video:
                image = video[key]
                break
        if image is None:
            if len(video) == 1:
                image = next(iter(video.values()))
            else:
                raise KeyError(f"no ego_view in obs['video']; keys={list(video)}")
        image = np.asarray(image)
        while image.ndim > 3 and image.shape[0] >= 1:   # peel [B, T, ...] to the newest frame
            image = image[-1]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"ego_view frame must be [H, W, 3], got {image.shape}")
        return image.astype(np.uint8)

    @staticmethod
    def _extract_proprio(obs: dict) -> np.ndarray:
        q = obs.get("q")
        if q is not None:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if q.shape[0] != 43:
                raise ValueError(f"obs['q'] must be 43-dim, got {q.shape[0]}")
            return q
        return state_from_groups(obs.get("state") or {})


def serve(policy: XR1GrootPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[xr1] PolicyServer ready on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running = False
                resp = {"status": "killing"}
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                resp = policy.get_action(
                    observation=data.get("observation", {}), options=data.get("options")
                )
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            resp = {"error": f"{type(error).__name__}: {error}"}
        try:
            sock.send(mnp.packb(resp, use_bin_type=True))
        except Exception:
            traceback.print_exc()
            break
    print("[xr1] server shut down", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True,
                        help="training run dir (holds config.py and last.ckpt/)")
    parser.add_argument("--ckpt-name", default="last.ckpt",
                        help="checkpoint dir inside --run-dir (default: last.ckpt)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--action-horizon", type=int, default=None,
                        help="truncate the returned chunk to this many steps")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="flow-ODE integration steps (model default 5); more steps "
                             "reduce the under-dispersion of the sampled chunk")
    parser.add_argument("--chunk-gain", type=float, default=1.0,
                        help="scale the chunk's temporal variation about its own mean "
                             "(1.0 = off). Deployment workaround for under-dispersion.")
    parser.add_argument("--prefix-len", type=int, default=0,
                        help="async inference: feed back this many steps of the previous "
                             "chunk as the action prefix (0 = plain chunked inference)")
    parser.add_argument("--replan-steps", type=int, default=20,
                        help="control steps the client consumes per chunk; the prefix is "
                             "taken from the previous chunk starting at this index "
                             "(50 Hz control / 2.5 Hz inference = 20)")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    policy = XR1GrootPolicy(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        ckpt_name=args.ckpt_name,
        device=args.device,
        initial_prompt=args.initial_prompt,
        exec_horizon=args.action_horizon,
        prefix_len=args.prefix_len,
        replan_steps=args.replan_steps,
        num_steps=args.num_steps,
        chunk_gain=args.chunk_gain,
    )
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
