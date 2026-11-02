"""CPU checks for previous converter entry points; no policy packages required."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [('xr1/scripts/data/soma_sonic_to_xr1_json.py', 'xr1/scripts/data/locomanilab_sonic_to_xr1_json.py')]


@pytest.mark.parametrize("old, canonical", ENTRY_POINTS)
def test_previous_cli_forwards_arguments_and_exit_code(tmp_path, old, canonical):
    previous = tmp_path / Path(old).name
    target = tmp_path / Path(canonical).name
    shutil.copy2(ROOT / old, previous)
    target.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    print(json.dumps([Path(__file__).name, sys.argv[1:]]))\n"
        "    raise SystemExit(7)\n"
    )
    result = subprocess.run(
        [sys.executable, str(previous), "--example", "value with spaces"],
        cwd=tmp_path.parent, text=True, capture_output=True,
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == [target.name, ["--example", "value with spaces"]]
    assert target.name in result.stderr


@pytest.mark.parametrize("old, canonical", ENTRY_POINTS)
def test_previous_import_reuses_the_canonical_module(tmp_path, old, canonical):
    previous = tmp_path / Path(old).name
    target = tmp_path / Path(canonical).name
    shutil.copy2(ROOT / old, previous)
    target.write_text("class Converter: pass\ndef main(): return 9\n")
    code = (
        "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
        "previous = importlib.import_module(sys.argv[2]); "
        "canonical = importlib.import_module(sys.argv[3]); "
        "assert previous.Converter is canonical.Converter; "
        "assert previous.main is canonical.main; "
        "assert previous.main() == 9"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), previous.stem, target.stem],
        cwd=tmp_path.parent, check=True,
    )


@pytest.mark.parametrize("explicit", [None, "soma_existing", "locomanilab_existing"])
def test_stats_name_defaults_to_new_prefix_and_preserves_explicit_names(tmp_path, monkeypatch, explicit):
    import importlib.util
    from types import ModuleType

    pytest.importorskip("numpy")
    sonic = ModuleType("mibot.utils.sonic")
    sonic.ACTION_DIM_SONIC = 80
    sonic.PROPRIO_DIM_SONIC = 43
    sonic.ACTION_PARTS_SONIC = [
        ("motion_token", slice(0, 64)),
        ("left_hand_joints", slice(64, 71)),
        ("right_hand_joints", slice(71, 78)),
    ]
    io = ModuleType("mibot.utils.io")
    io.STATE_DIM = 60
    monkeypatch.setitem(sys.modules, sonic.__name__, sonic)
    monkeypatch.setitem(sys.modules, io.__name__, io)
    script = ROOT / "xr1/scripts/data/compute_xr1_stats.py"
    spec = importlib.util.spec_from_file_location("local_xr1_stats", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "demo-task v1"
    source.mkdir()
    episode = {
        "actions": {
            "motion_token": [[0.0] * 64] * 2,
            "left_hand_joints": [[0.0] * 7] * 2,
            "right_hand_joints": [[0.0] * 7] * 2,
        },
        "proprios": {"joint_pos": [[0.0] * 43, [1.0] * 43]},
    }
    (source / "episode.json").write_text(json.dumps(episode))
    output = tmp_path / "configs"
    arguments = [str(script), "--json-dir", str(source), "--config-dir", str(output)]
    if explicit is not None:
        arguments += ["--name", explicit]
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    name = explicit if explicit is not None else "locomanilab_demo_task_v1"
    assert {path.name for path in output.iterdir()} == {name + ".yaml"}
    text = (output / (name + ".yaml")).read_text()
    assert str(source.resolve()) in text
    assert "type: SonicDataModule" in text
    assert "mean:" in text and "q99:" in text
