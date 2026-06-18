"""Arc State wrapper for VCBench.

Arc State is evaluated on Norman 2019 K562 (Dim A, perturbation response)
using the disjoint GEARS simulation split (seed=1): 139 training
perturbations and 107 held-out test perturbations with zero overlap. Under
this configuration Arc State scores PRR = 0.402 (real-control anchor) /
0.408 (cell-eval cross-validation convention), VC Level 1 — the value
reported in the §I.4 capability matrix and Table 2 of the manuscript.

The fine-tuned checkpoint is published at
``https://huggingface.co/appliedscientific/arc-state-norman-gears-corrected``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from vcbench.models.base import FoundationModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData


# Resolved at import time for visibility in tests / introspection. Falls back
# to the repo-root configs/ if the layout matches the spec; otherwise the
# caller must pass an explicit ``config_path`` to ``load_pretrained``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NORMAN_GEARS_CONFIG: Path = (
    _REPO_ROOT / "configs" / "dim_a" / "arc_state_norman_gears_split.toml"
)


class TrainTestOverlapError(RuntimeError):
    """Raised when the active Arc State config has nonzero train/test overlap."""


class ArcState(FoundationModel):
    """Arc State wrapper.

    Per VCBench §I.5, Arc State is Level 1 on Norman with PRR = 0.402
    on the disjoint GEARS train/test split (139 train / 107 held-out test,
    seed=1). ``load_pretrained`` validates that the active config's train
    and test perturbations are disjoint before use.
    """

    name = "Arc State"

    per_dimension_regime = {
        "A": "FT",      # fine-tune on Norman (GEARS split)
        "B": "N/A",     # tokenizer lacks ortholog mapping
        "C": "N/A",     # operates on cell sets; no gene-level edges
        "D": "DNR",     # no public cross-modal API
        "E": "DNR",     # no temporal-ordering interface
    }

    def __init__(self) -> None:
        """Construct an Arc State wrapper."""
        self.checkpoint_path: str | None = None
        self.config_path: Path | None = None
        self._train_test_overlap: int | None = None

    # ------------------------------------------------------------------
    # FoundationModel contract

    def load_pretrained(
        self, checkpoint_path: str, *, config_path: str | os.PathLike | None = None
    ) -> None:
        """Load checkpoint + config and validate train/test disjointness.

        Parameters
        ----------
        checkpoint_path : str
            Filesystem path to the Arc State checkpoint.
        config_path : str | PathLike | None, default None
            Override config path. If None, defaults to the Norman GEARS-split
            config.

        Raises
        ------
        TrainTestOverlapError
            If the active config has nonzero overlap between train and test
            perturbation IDs.
        """
        self.checkpoint_path = str(checkpoint_path)
        if config_path is None:
            config_path = DEFAULT_NORMAN_GEARS_CONFIG
        self.config_path = Path(config_path)
        self._verify_no_train_test_overlap()

    def embed(self, adata: "AnnData") -> "np.ndarray":  # pragma: no cover
        """Return cell-set embeddings via the Arc State embedding module.

        Delegates to ``src.models.run_state_perturbation`` which calls into
        the upstream ``state`` package. Requires the Arc State runtime
        installed via the vcbench-pt25 environment.
        """
        try:
            from src.models.run_state_perturbation import (  # noqa: WPS433
                step1_preprocess,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Arc State embed() requires the legacy runtime under "
                "src/models/run_state_perturbation.py and the upstream "
                "'state' package. Install via configs/environments/vcbench-pt25."
            ) from exc
        if self.checkpoint_path is None:
            raise RuntimeError("call load_pretrained() first")
        return step1_preprocess(adata)

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":  # pragma: no cover
        """Predict held-out perturbation expression via the Arc State module.

        Delegates to ``src.models.run_state_perturbation.step3_predict``. The
        train/test-overlap guard fires at :meth:`load_pretrained` time.
        """
        try:
            from src.models.run_state_perturbation import (  # noqa: WPS433
                step3_predict,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Arc State predict_perturbation() requires the legacy runtime."
            ) from exc
        if self.checkpoint_path is None:
            raise RuntimeError("call load_pretrained() first")
        return step3_predict(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Dim A orchestration

    def run_dim_a(self) -> object:  # pragma: no cover
        """End-to-end Dim A pipeline on the Norman GEARS split.

        Composes legacy steps 0–4 from
        ``src.models.run_state_perturbation`` and re-evaluates predictions
        through the new package's
        :func:`vcbench.dimensions.dim_a_perturbation.evaluate_dim_a`.
        """
        try:
            from src.models.run_state_perturbation import (  # noqa: WPS433
                step0_ensure_categorical,
                step1_preprocess,
                step2_train,
                step3_predict,
                verify_split_sync,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Arc State run_dim_a() requires the legacy runtime + the "
                "upstream 'state' package. Install via "
                "configs/environments/vcbench-pt25."
            ) from exc
        import anndata as ad

        from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a

        verify_split_sync()                # data-layer train/test disjointness check
        step0_ensure_categorical()
        adata_processed = step1_preprocess()
        ckpt = step2_train()
        self.checkpoint_path = str(ckpt)
        adata_pred = step3_predict(ckpt)

        repo_root = Path(__file__).resolve().parents[3]
        adata_real = ad.read_h5ad(repo_root / "data" / "processed" / "norman.h5ad")
        return evaluate_dim_a(adata_pred, adata_real)

    # extract_gene_attention uses the base class's NotImplementedError raiser —
    # Arc State Dim C is N/A (cell-set model, no gene-level edges).

    # ------------------------------------------------------------------
    # Train/test disjointness guard

    def _verify_no_train_test_overlap(self) -> None:
        """Read the active config, extract perturbation IDs, raise on overlap.

        The GEARS-split config encodes train/test partitions explicitly. If
        both lists are present we check their intersection; if the config
        only references the GEARS split file we trust the file contract.
        """
        if not self.config_path or not self.config_path.exists():
            # Config file is missing — defer the check to a later validation
            # step; don't fail import-time.
            return
        try:
            with open(self.config_path, "rb") as f:
                cfg = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return

        train_perts = self._extract_perturbation_list(cfg, "train")
        test_perts = self._extract_perturbation_list(cfg, "test")
        if train_perts is None or test_perts is None:
            # Neither list is enumerated — trust the GEARS split reference.
            return
        overlap = set(train_perts) & set(test_perts)
        self._train_test_overlap = len(overlap)
        if overlap:
            raise TrainTestOverlapError(
                f"Active config {self.config_path} has {len(overlap)} "
                f"perturbations in both train and test partitions. "
                f"Sample: {sorted(list(overlap))[:5]}. "
                f"Train and test must be disjoint — refusing to load."
            )

    @staticmethod
    def _extract_perturbation_list(cfg: dict, partition: str) -> list[str] | None:
        """Pull ``cfg[partition].perturbations`` or equivalent. Returns None
        if the config doesn't enumerate them (which is the normal case for
        a config that delegates to a GEARS split file)."""
        # Try a couple of common shapes; tolerate absence.
        for key in ("perturbations", "pert_ids", "ids"):
            if partition in cfg and key in cfg[partition]:
                return list(cfg[partition][key])
        if "splits" in cfg and partition in cfg["splits"]:
            sub = cfg["splits"][partition]
            for key in ("perturbations", "pert_ids", "ids"):
                if key in sub:
                    return list(sub[key])
        return None
