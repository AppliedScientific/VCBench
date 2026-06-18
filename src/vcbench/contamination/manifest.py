"""Contamination manifest schema and validator.

A ``ContaminationManifest`` declares, for one foundation-model release:

* the model identifier and version,
* the canonical pretraining corpus (with optional MD5 of the cell-barcode list),
* the explicit ``included`` and ``excluded`` accessions / cell types / tissues
  the trainers asserted (e.g. Geneformer's exclusion of immortalised lines),
* the schema version string so future revisions are detectable.

``validate_manifest()`` parses a YAML / JSON / Python-dict manifest and returns
a :class:`ValidationSummary` (success path) or raises
:class:`ManifestValidationError` (failure path) listing every constraint that
was violated.

The schema deliberately models *what the trainers documented* rather than what
can be empirically verified — empirical verification requires the cell-barcode
manifest, which no public single-cell FM release ships at the time of writing.
``schema_version: vcbench-contamination-v1`` is the closest existing single-cell
analogue to the "is-it-in-my-training-set" disclosure expected of LLM releases.
"""

from __future__ import annotations

import json
import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ManifestValidationError(Exception):
    """Raised when a manifest fails validation. Carries a list of error strings."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        joined = "\n  - " + "\n  - ".join(self.errors) if self.errors else ""
        super().__init__(f"manifest validation failed:{joined}")


class ContaminationManifest(BaseModel):
    """Pretraining-overlap declaration for one foundation-model release.

    Attributes
    ----------
    schema_version : str
        Schema version tag. Must equal ``"vcbench-contamination-v1"`` for
        manifests intended to be consumed by ``validate_manifest``.
    model : str
        Canonical model identifier (e.g. ``"geneformer-v2-316m"``).
    model_version : str | None
        Optional sub-version / checkpoint tag.
    pretraining_corpus : str
        Human-readable corpus name (e.g. ``"Genecorpus-30M"``).
    pretraining_corpus_md5 : str | None
        Optional MD5 of the cell-barcode list (for releases that ship one).
    included_datasets : list[str]
        Accessions / dataset names the trainers asserted are *in* the corpus.
        May be empty if the trainers did not enumerate inclusions.
    excluded_datasets : list[str]
        Accessions / dataset names the trainers asserted are *out* of the corpus
        (e.g. Geneformer excluding immortalised cell lines).
    excluded_tissue_types : list[str]
        High-level filters such as ``["cell culture", "perturbation_assay"]``
        (Census-style structural exclusions).
    notes : str | None
        Free-form provenance comments.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(default="vcbench-contamination-v1")
    model: str
    model_version: str | None = None
    pretraining_corpus: str
    pretraining_corpus_md5: str | None = None
    included_datasets: list[str] = Field(default_factory=list)
    excluded_datasets: list[str] = Field(default_factory=list)
    excluded_tissue_types: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_schema(cls, v: str) -> str:
        if v != "vcbench-contamination-v1":
            raise ValueError(
                "schema_version must be 'vcbench-contamination-v1' "
                f"(got {v!r})"
            )
        return v


class ValidationSummary(BaseModel):
    """Successful-validation summary returned by :func:`validate_manifest`."""

    model_config = ConfigDict(extra="forbid")

    model: str
    pretraining_corpus: str
    n_included: int
    n_excluded: int
    n_excluded_tissue_types: int
    has_md5: bool
    schema_version: str
    warnings: list[str] = Field(default_factory=list)


def _load_dict(source: str | os.PathLike[str] | dict[str, Any]) -> dict[str, Any]:
    """Load a manifest from a YAML/JSON path or accept an in-memory dict."""
    if isinstance(source, dict):
        return dict(source)
    path = os.fspath(source)
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        loaded = yaml.safe_load(text)
    elif path.endswith(".json"):
        loaded = json.loads(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError:
            loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ManifestValidationError(
            [f"manifest must deserialise to a dict, got {type(loaded).__name__}"]
        )
    return loaded


def validate_manifest(
    source: str | os.PathLike[str] | dict[str, Any],
) -> ValidationSummary:
    """Validate a contamination manifest.

    Parameters
    ----------
    source : str | PathLike | dict
        A path to a YAML/JSON manifest file or an in-memory dict.

    Returns
    -------
    ValidationSummary
        On success. The ``warnings`` list captures non-fatal issues
        (e.g. missing MD5, both included and excluded lists empty).

    Raises
    ------
    ManifestValidationError
        If parsing or schema validation fails. The ``errors`` attribute
        holds every validation message.

    Examples
    --------
    >>> m = {
    ...     "schema_version": "vcbench-contamination-v1",
    ...     "model": "geneformer-v2-316m",
    ...     "pretraining_corpus": "Genecorpus-30M",
    ...     "excluded_datasets": ["GSE133344"],
    ... }
    >>> summary = validate_manifest(m)
    >>> summary.model
    'geneformer-v2-316m'
    >>> summary.n_excluded
    1
    """
    raw = _load_dict(source)
    try:
        manifest = ContaminationManifest(**raw)
    except Exception as exc:
        raise ManifestValidationError([str(exc)]) from exc

    warnings: list[str] = []
    if manifest.pretraining_corpus_md5 is None:
        warnings.append(
            "no pretraining_corpus_md5 provided — empirical contamination "
            "checks against new evaluation datasets cannot be computed"
        )
    if not manifest.included_datasets and not manifest.excluded_datasets:
        warnings.append(
            "neither included_datasets nor excluded_datasets populated — "
            "manifest will only support structural (tissue-type-level) checks"
        )
    return ValidationSummary(
        model=manifest.model,
        pretraining_corpus=manifest.pretraining_corpus,
        n_included=len(manifest.included_datasets),
        n_excluded=len(manifest.excluded_datasets),
        n_excluded_tissue_types=len(manifest.excluded_tissue_types),
        has_md5=manifest.pretraining_corpus_md5 is not None,
        schema_version=manifest.schema_version,
        warnings=warnings,
    )
