"""``python -m vcbench`` CLI entry point.

Subcommands::

    python -m vcbench tests                  # Run the unit test suite
    python -m vcbench dim_a [--model M]      # Run Dim A pipeline (or one model)
    python -m vcbench dim_b [--model M]
    python -m vcbench dim_c
    python -m vcbench dim_d
    python -m vcbench dim_e
    python -m vcbench levels                 # Print VC Level assignments
    python -m vcbench audit-contamination    # Validate manifests + cross-check
    python -m vcbench show-config            # Dump pre_registration.yaml
    python -m vcbench drift                  # Run reference-value drift detectors

Models accepted by ``--model``: geneformer, scgpt, uce, transcriptformer, arc_state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from vcbench import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# Subcommand handlers


def _cmd_tests(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _cmd_drift(args: argparse.Namespace) -> int:
    """Run only the reference-value drift detectors."""
    cmd = [
        sys.executable, "-m", "pytest", "tests/unit/", "-q", "-k",
        "matches_reference or invariant or partition_decomposition or weinreb",
    ]
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _resolve_model(name: str):
    name = name.lower().replace("-", "_")
    from vcbench.models import (
        SCGPT,
        UCE,
        ArcState,
        Geneformer,
        TranscriptFormer,
    )
    table = {
        "geneformer": Geneformer,
        "scgpt": SCGPT,
        "uce": UCE,
        "transcriptformer": TranscriptFormer,
        "arc_state": ArcState,
        "arc-state": ArcState,
        "state": ArcState,
    }
    if name not in table:
        raise SystemExit(
            f"unknown model {name!r}; choose from "
            f"{sorted(set(table.keys()))}"
        )
    return table[name]()


def _cmd_dim_a(args: argparse.Namespace) -> int:
    if args.model is None:
        # No model = run every model that supports Dim A.
        for m in ["geneformer", "scgpt", "transcriptformer", "arc_state"]:
            print(f"=== Dim A: {m} ===")
            _run_one_dim_a(_resolve_model(m))
        return 0
    return _run_one_dim_a(_resolve_model(args.model))


def _run_one_dim_a(model) -> int:
    if not model.is_dimension_supported("A"):
        print(
            f"{model.name}: Dim A regime is "
            f"{model.regime_for('A')!r} — skipping."
        )
        return 0
    if not hasattr(model, "run_dim_a"):
        print(
            f"{model.name}: no run_dim_a() entry point yet."
        )
        return 0
    try:
        result = model.run_dim_a()
    except NotImplementedError as exc:
        print(f"{model.name}: {exc}")
        return 0
    print(f"{model.name} Dim A:")
    if hasattr(result, "to_aggregate_dict"):
        print(json.dumps(result.to_aggregate_dict(), indent=2, default=str))
    else:
        print(result)
    return 0


def _cmd_dim_b(args: argparse.Namespace) -> int:
    print("Dim B per-tissue dual-protocol evaluation runs via the pipeline "
          "(src/models/); reference outputs are in results/dim_b/.")
    print(
        "Reference values are accessible immediately via:\n"
        "  python -c 'import json; "
        "print(json.dumps(json.load(open(\"tests/reference_values.json\"))[\"dim_b\"], indent=2))'"
    )
    return 0


def _cmd_dim_c(args: argparse.Namespace) -> int:
    print("Dim C bootstrap CIs run via the pipeline (src/models/); "
          "reference outputs are in results/dim_c/.")
    return 0


def _cmd_dim_d(args: argparse.Namespace) -> int:
    print("Dim D cross-modal probe runs via the pipeline (src/models/); "
          "reference outputs are in results/dim_d/.")
    return 0


def _cmd_dim_e(args: argparse.Namespace) -> int:
    print("Dim E temporal ordering runs via the pipeline (src/models/); "
          "reference outputs are in results/dim_e/.")
    return 0


def _cmd_levels(args: argparse.Namespace) -> int:
    """Print VC Level assignments from the pre-registration."""
    import yaml
    pre_reg = REPO_ROOT / "configs" / "pre_registration.yaml"
    cfg = yaml.safe_load(pre_reg.read_text())
    print("VC Level assignments (pre-registered, §I.5):\n")
    for model, level in cfg["expected_assignments"].items():
        print(f"  {model:<22} → Level {level}")
    print()
    print(f"VC Level definitions:")
    for k, v in cfg["vc_levels"].items():
        print(f"  {k}: {v}")
    return 0


def _cmd_audit_contamination(args: argparse.Namespace) -> int:
    """Validate every shipped contamination manifest example."""
    from vcbench.contamination import validate_manifest
    examples_dir = (
        Path(__file__).resolve().parent / "contamination" / "examples"
    )
    failures = 0
    for path in sorted(examples_dir.glob("*.yaml")):
        try:
            summary = validate_manifest(path)
            print(f"  ✓ {path.name}: {summary.model} (corpus={summary.pretraining_corpus})")
            for w in summary.warnings:
                print(f"      warn: {w}")
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}")
            failures += 1
    return 1 if failures else 0


def _cmd_show_config(args: argparse.Namespace) -> int:
    pre_reg = REPO_ROOT / "configs" / "pre_registration.yaml"
    print(pre_reg.read_text())
    return 0


# ----------------------------------------------------------------------
# Argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vcbench",
        description=f"VCBench v{__version__} — single-cell foundation-model benchmark.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("tests", help="Run the unit test suite (no GPU)")
    sub.add_parser("drift", help="Run reference-value drift detectors only")

    pa = sub.add_parser("dim_a", help="Run Dim A (perturbation) pipeline")
    pa.add_argument("--model", default=None,
                    help="Run only one model (geneformer / scgpt / arc_state / "
                         "transcriptformer / uce). Default: every Dim-A-supporting model.")

    sub.add_parser("dim_b", help="Run Dim B (cross-species) — see help text")
    sub.add_parser("dim_c", help="Run Dim C (GRN) — see help text")
    sub.add_parser("dim_d", help="Run Dim D (cross-modal) — see help text")
    sub.add_parser("dim_e", help="Run Dim E (temporal) — see help text")

    sub.add_parser("levels", help="Print VC Level assignments from pre-registration")
    sub.add_parser("audit-contamination",
                   help="Validate shipped contamination manifests")
    sub.add_parser("show-config",
                   help="Dump configs/pre_registration.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = {
        "tests":               _cmd_tests,
        "drift":               _cmd_drift,
        "dim_a":               _cmd_dim_a,
        "dim_b":               _cmd_dim_b,
        "dim_c":               _cmd_dim_c,
        "dim_d":               _cmd_dim_d,
        "dim_e":               _cmd_dim_e,
        "levels":              _cmd_levels,
        "audit-contamination": _cmd_audit_contamination,
        "show-config":         _cmd_show_config,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
