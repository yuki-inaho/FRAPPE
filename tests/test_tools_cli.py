"""Every tool must import and present its interface.

The tools used to import each other -- ``export_onnx`` took its checkpoint
loader from ``evaluate_joint_prefix`` and its image loader from
``prune_latent_channels`` -- so renaming a helper in one tool broke three others,
and nothing noticed until someone ran them. They now share a library instead, and
this suite is what keeps the import graph honest: it costs a second and it fails
the moment a tool stops being runnable.

``--help`` is the cheapest end-to-end check there is. Reaching it means the
module imported, every top-level statement ran, and the parser was built.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted(path.name for path in (REPOSITORY_ROOT / "tools").glob("*.py")
               if path.name != "__init__.py")
ENTRY_POINTS = ["train_joint_prefix.py", "train_managed.py", "train_rae_progressive.py"]


@pytest.mark.parametrize("script", TOOLS + ENTRY_POINTS)
def test_script_builds_its_command_line(script: str) -> None:
    target = REPOSITORY_ROOT / ("tools/" + script if script in TOOLS else script)
    result = subprocess.run([sys.executable, str(target), "--help"],
                            capture_output=True, text=True, timeout=180,
                            cwd=REPOSITORY_ROOT, check=False)
    assert result.returncode == 0, (
        f"{script} --help failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    assert result.stdout.strip(), f"{script} --help printed nothing"


def test_no_tool_imports_another_tool() -> None:
    """Shared code belongs in the harness, not in whichever tool wrote it first.

    A tool importing a tool makes the earlier one a library it was never designed
    to be: its helpers acquire callers it cannot see, and its command-line
    scaffolding is imported for nothing.
    """
    offenders = {}
    for path in (REPOSITORY_ROOT / "tools").glob("*.py"):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        borrowed = [line for line in lines
                    if line.startswith(("from tools.", "import tools."))]
        if borrowed:
            offenders[path.name] = borrowed
    assert not offenders, f"tools importing tools: {offenders}"


def test_the_dataset_root_has_one_definition() -> None:
    """A default path repeated in nine files is nine places to forget.

    ``tests/`` may name it: this file searches for it, and the managed-config
    test asserts what the Hydra data config resolves to. Nothing under ``tools/``
    or at the repository root may, because those are the ones that get missed.
    """
    literal = "frappe_rgb_800x608/imagefolder"
    holders = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in REPOSITORY_ROOT.glob("**/*.py")
        if ".pixi" not in path.parts and ".venv" not in path.parts
        and literal in path.read_text(encoding="utf-8"))
    allowed = {"src/compressors/frappe/harness/data.py",
               "tests/test_tools_cli.py", "tests/test_managed_config.py"}
    assert set(holders) <= allowed, sorted(set(holders) - allowed)
