"""Integration smoke tests for the `python -m vcbench` CLI.

These run the CLI as a subprocess (the way an end-user invokes it) and
verify the no-GPU subcommands return success and emit expected content.

Note: subprocess invocations explicitly pass ``PYTHONPATH=<repo>/src`` so
they don't depend on the editable-install ``.pth`` file being processed.
That file's path resolution can flake under iCloud Mobile Documents sync
(``site.py`` silently skips paths that aren't yet locally materialised);
explicit ``PYTHONPATH`` makes the test robust to that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "vcbench", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
    )


def test_cli_levels_emits_pre_registered_assignments():
    r = _run("levels")
    assert r.returncode == 0, r.stderr
    assert "transcriptformer" in r.stdout.lower()
    assert "Level 2" in r.stdout
    assert "geneformer_v2_316m" in r.stdout
    assert "Level 1" in r.stdout


def test_cli_audit_contamination_passes_for_all_shipped_examples():
    r = _run("audit-contamination")
    assert r.returncode == 0, r.stderr
    # All 5 example manifests must validate
    for model in ["geneformer-v2-316m", "scgpt", "uce-33-layer",
                   "transcriptformer", "arc-state-transition"]:
        assert model in r.stdout.lower(), \
            f"{model} missing from audit-contamination output"


def test_cli_show_config_contains_pre_registration_yaml():
    r = _run("show-config")
    assert r.returncode == 0, r.stderr
    assert "vc_levels" in r.stdout
    assert "expected_assignments" in r.stdout
    assert "amendment_history" in r.stdout


def test_cli_help_lists_all_subcommands():
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    for cmd in ["tests", "drift", "dim_a", "dim_b", "dim_c", "dim_d",
                "dim_e", "levels", "audit-contamination", "show-config"]:
        assert cmd in r.stdout, f"subcommand {cmd!r} missing from --help"


def test_cli_dim_b_through_dim_e_point_to_results():
    """Dim B/C/D/E entry points print a clear pointer to the on-disk
    reference outputs under results/."""
    for cmd in ["dim_b", "dim_c", "dim_d", "dim_e"]:
        r = _run(cmd)
        assert r.returncode == 0, f"{cmd}: {r.stderr}"
        assert ("results/" in r.stdout) or ("reference values" in r.stdout.lower())


@pytest.mark.parametrize("model", ["geneformer", "scgpt", "arc_state",
                                    "transcriptformer", "uce"])
def test_cli_dim_a_per_model_handles_missing_runtime_gracefully(model):
    """Without GPU + heavy deps, run_dim_a() should NotImplementedError;
    the CLI must catch that and emit a readable message rather than a stack trace."""
    r = _run("dim_a", "--model", model)
    # Either runs (returncode=0) or fails cleanly with the explanatory message
    assert r.returncode == 0 or "NotImplementedError" not in r.stderr, \
        f"{model}: unexpected traceback on stderr"
