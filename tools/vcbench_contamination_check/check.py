# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 VCBench authors
"""
Reference implementation for the VCBench Contamination Reporting Schema v1.

Four possible verdicts per (model, evaluation-dataset) pair:
  * confirmed  — non-empty MD5 intersection between pretraining manifest and
                 evaluation barcodes
  * likely     — no MD5 intersection, but evaluation accession is listed as
                 INCLUDED in pretraining_manifest.yaml
  * unlikely   — no MD5 intersection, accession listed as EXCLUDED
  * unknown    — no MD5 intersection, accession not mentioned in manifest

Plus an auxiliary verdict for schema non-compliance:
  * schema_incomplete — required manifest files missing

See docs/contamination-schema-v1.md for the full specification.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import click

logger = logging.getLogger(__name__)


# ── Required files ──────────────────────────────────────────────────────

MD5_FILE_PARQUET = "training_cells.md5.parquet"
MD5_FILE_TXT = "training_cells.md5.txt"
MANIFEST_FILE = "pretraining_manifest.yaml"


def _find_md5_file(model_release_dir: Path) -> Path | None:
    for name in (MD5_FILE_PARQUET, MD5_FILE_TXT):
        p = model_release_dir / name
        if p.exists():
            return p
    return None


def _load_pretrain_md5(md5_path: Path) -> set[str]:
    if md5_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(md5_path)
        if "barcode_md5" not in df.columns:
            raise ValueError(
                f"{md5_path}: parquet file missing required 'barcode_md5' column"
            )
        return set(df["barcode_md5"].astype(str))
    else:  # .txt
        lines = [
            line.strip().split("\t")[0]
            for line in md5_path.read_text().splitlines()
            if line.strip()
        ]
        return set(lines)


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(manifest_path.read_text())


def _hash_eval_barcodes(adata_path: Path) -> tuple[set[str], str | None]:
    """Return (md5 set, adata.uns.get('source_accession'))."""
    import anndata as ad
    a = ad.read_h5ad(adata_path)
    barcodes = a.obs_names.astype(str).tolist()
    md5s = {hashlib.md5(b.encode("utf-8")).hexdigest() for b in barcodes}
    accession = a.uns.get("source_accession")
    return md5s, accession


def check_contamination(
    model_release_dir: str | Path,
    eval_dataset: str | Path,
) -> Dict[str, Any]:
    """Run the verdict pipeline on one (model release, eval dataset) pair."""
    model_release_dir = Path(model_release_dir)
    eval_dataset = Path(eval_dataset)

    # 1. Schema compliance
    md5_file = _find_md5_file(model_release_dir)
    manifest_file = model_release_dir / MANIFEST_FILE
    missing = []
    if md5_file is None:
        missing.append(str(model_release_dir / MD5_FILE_PARQUET) + " (or .txt)")
    if not manifest_file.exists():
        missing.append(str(manifest_file))
    if missing:
        return {
            "verdict":      "schema_incomplete",
            "missing":      missing,
            "model_dir":    str(model_release_dir),
            "eval_dataset": str(eval_dataset),
        }

    # 2. MD5 intersection
    pretrain_md5 = _load_pretrain_md5(md5_file)
    eval_md5, eval_accession = _hash_eval_barcodes(eval_dataset)
    intersection = pretrain_md5 & eval_md5
    if intersection:
        return {
            "verdict":          "confirmed",
            "overlap_cells":    len(intersection),
            "overlap_fraction": len(intersection) / len(eval_md5) if eval_md5 else 0.0,
            "model_dir":        str(model_release_dir),
            "eval_dataset":     str(eval_dataset),
        }

    # 3. Accession lookup
    manifest = _load_manifest(manifest_file)
    if eval_accession is None:
        return {
            "verdict":      "unknown",
            "reason":       f"{eval_dataset} has no source_accession in adata.uns",
            "model_dir":    str(model_release_dir),
            "eval_dataset": str(eval_dataset),
        }

    sources = manifest.get("sources") or []
    matched = [s for s in sources if s.get("accession") == eval_accession]
    if not matched:
        return {
            "verdict":      "unknown",
            "reason":       f"{eval_accession} not listed in {MANIFEST_FILE}",
            "model_dir":    str(model_release_dir),
            "eval_dataset": str(eval_dataset),
        }

    entry = matched[0]
    if entry.get("included"):
        return {
            "verdict":      "likely",
            "rationale":    entry.get("notes", ""),
            "accession":    eval_accession,
            "model_dir":    str(model_release_dir),
            "eval_dataset": str(eval_dataset),
        }
    return {
        "verdict":      "unlikely",
        "exclusions":   entry.get("exclusions", []),
        "rationale":    entry.get("notes", ""),
        "accession":    eval_accession,
        "model_dir":    str(model_release_dir),
        "eval_dataset": str(eval_dataset),
    }


# ── CLI ─────────────────────────────────────────────────────────────────

@click.command("vcbench-check-contamination")
@click.argument("model_release_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("eval_dataset", type=click.Path(exists=True, dir_okay=False))
@click.option("--json-out", type=click.Path(dir_okay=False),
              help="Optional path to write the verdict as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(model_release_dir: str, eval_dataset: str, json_out: str | None, verbose: bool):
    """Check training-data contamination between a released model and an eval dataset.

    Prints a verdict: confirmed / likely / unlikely / unknown / schema_incomplete.
    Exit code 0 for unlikely or unknown; 1 for confirmed; 2 for likely; 3 for
    schema_incomplete.
    """
    import json

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    verdict_dict = check_contamination(model_release_dir, eval_dataset)
    verdict = verdict_dict["verdict"]

    click.echo(f"verdict: {verdict}")
    for k, v in verdict_dict.items():
        if k == "verdict":
            continue
        click.echo(f"  {k}: {v}")

    if json_out:
        Path(json_out).write_text(json.dumps(verdict_dict, indent=2, default=str))

    exit_codes = {
        "confirmed":          1,
        "likely":              2,
        "unlikely":            0,
        "unknown":             0,
        "schema_incomplete":   3,
    }
    sys.exit(exit_codes.get(verdict, 0))


if __name__ == "__main__":
    cli()
