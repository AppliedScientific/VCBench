"""scGPT (fine-tuned) wrapper.

Per §I.4: A=FT, B=ZS, C=IE, D=ZS+D, E=ZS.

Dim A pipeline (Cui et al. 2024 scGPT perturbation tutorial):

1. Load Norman via GEARS PertData with the simulation split (seed=1).
2. Build vocab + ``TransformerGenerator`` from the released ``scGPT_human``
   checkpoint.
3. Fine-tune with masked MSE loss: up to 15 epochs, early stopping
   patience 10, lr 1e-4, batch 16, StepLR γ=0.9, max_seq_len 1536.
4. Predict held-out perturbations via the native ``pred_perturb`` API.
5. Evaluate via the new
   :func:`vcbench.dimensions.dim_a_perturbation.evaluate_dim_a`.

Reference: scGPT FT PRR on Norman full-test = 0.5025 (§I.4 Table 2).
Per-partition: shared 71-doubles 0.5445, novel 36-singles 0.4196.

Engineering note (Bo Wang Lab):
This wrapper composes verified step functions from
``src.models.run_scgpt_perturbation`` rather than re-implementing the
masked-MSE training loop. The fine-tune that produced the §I.4
reference values follows the documented recipe (training recipe +
checkpoint provenance).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vcbench.models.base import FoundationModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData

    from vcbench.dimensions.dim_a_perturbation.evaluate import DimAResult


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR: Path = _REPO_ROOT / "models" / "scgpt" / "scGPT_human"
DEFAULT_RESULTS_DIR: Path = _REPO_ROOT / "results" / "dim_a" / "scgpt"
DEFAULT_PROCESSED_DIR: Path = _REPO_ROOT / "data" / "processed"


@dataclass
class SCGPTConfig:
    """Runtime configuration for the scGPT wrapper.

    Defaults reproduce §I.2. Override only for tests / ablations.
    """

    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR

    # Dim A fine-tuning hyperparameters (§I.2)
    epochs_max: int = 15
    early_stopping_patience: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 16
    step_lr_gamma: float = 0.9
    max_seq_length: int = 1536
    seed: int = 42

    # GEARS split spec
    gears_split_kind: str = "simulation"
    gears_split_seed: int = 1


class SCGPT(FoundationModel):
    """scGPT (fine-tuned) wrapper."""

    name = "scGPT (fine-tuned)"

    per_dimension_regime = {
        "A": "FT",
        "B": "ZS",
        "C": "IE",
        "D": "ZS+D",
        "E": "ZS",
    }

    def __init__(self, config: SCGPTConfig | None = None) -> None:
        self.config = config or SCGPTConfig()
        self.checkpoint_path: str | None = None
        self._model: object | None = None
        self._gene_ids: object | None = None
        self._pert_data: object | None = None

    # ------------------------------------------------------------------
    # FoundationModel contract

    def load_pretrained(self, checkpoint_path: str | os.PathLike) -> None:
        """Pin the checkpoint path. Heavy load happens in :meth:`run_dim_a`."""
        self.checkpoint_path = str(checkpoint_path)
        self.config.checkpoint_dir = Path(checkpoint_path)

    def embed(self, adata: "AnnData") -> "np.ndarray":  # pragma: no cover
        """Return cell embeddings via the fine-tuned scGPT encoder.

        scGPT's native interface returns predictions rather than bare
        embeddings; for downstream Dim B/D/E use we expose the encoder
        output via the ``embsum_*`` utilities in the legacy module.
        """
        try:
            import scgpt  # noqa: F401  (Bo Wang Lab runtime, optional dep)
        except ImportError as exc:
            raise NotImplementedError(
                "scGPT embed() requires the scgpt Python package. "
                "Install via configs/environments/vcbench-scgpt-pinned.txt."
            ) from exc
        raise NotImplementedError(
            "scGPT bare-embedding extraction is in the legacy "
            "src/models/run_crossmodal_embeddings.py pipeline; the wrapper "
            "exposes only the perturbation-prediction surface in v1.0."
        )

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":  # pragma: no cover
        """Predict held-out perturbation expression via the fine-tuned model.

        Composes legacy step4_predict from
        ``src.models.run_scgpt_perturbation``. Requires :meth:`run_dim_a`
        (or :meth:`fine_tune`) to have been called first.
        """
        try:
            from src.models.run_scgpt_perturbation import (  # noqa: WPS433
                step4_predict,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "scGPT predict_perturbation() requires the legacy runtime "
                "under src/models/run_scgpt_perturbation.py."
            ) from exc
        if self._model is None or self._pert_data is None:
            raise RuntimeError("call run_dim_a() or fine_tune() first")
        return step4_predict(self._model, self._pert_data, self._gene_ids)

    def extract_gene_attention(self) -> "np.ndarray":  # pragma: no cover
        """Per-gene attention surface from the released checkpoint (regime IE).

        scGPT Dim C IE is implemented as gene-embedding cosine similarity
        from the released checkpoint, **without** task-specific fine-tuning
        or cell-conditioning (per §I.2). The runtime path lives in the
        pipeline GRN module (src/models/); reference outputs are in
        results/dim_c/scgpt/.
        """
        raise NotImplementedError(
            "scGPT GRN inference (Dim C IE) runs via the pipeline "
            "(src/models/); reference outputs are in results/dim_c/scgpt/. "
            "Implementation: gene-embedding cosine similarity from the "
            "released scGPT_human checkpoint, no task-specific fine-tuning "
            "(per §I.2 regime IE)."
        )

    # ------------------------------------------------------------------
    # Dim A orchestration

    def run_dim_a(self) -> "DimAResult":  # pragma: no cover
        """End-to-end Dim A pipeline. Returns a DimAResult.

        Composes legacy steps 1–5 from ``src.models.run_scgpt_perturbation``
        and re-evaluates predictions through the new package's
        :func:`vcbench.dimensions.dim_a_perturbation.evaluate_dim_a`.
        """
        try:
            from src.models.run_scgpt_perturbation import (  # noqa: WPS433
                step1_load_data,
                step2_build_vocab_and_model,
                step3_finetune,
                step4_predict,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "scGPT run_dim_a() requires the scgpt + GEARS runtime."
            ) from exc
        import anndata as ad

        from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a

        self._pert_data = step1_load_data()
        model, vocab, gene_ids = step2_build_vocab_and_model(self._pert_data)
        if model is None:
            raise RuntimeError("scGPT step2 returned None — checkpoint missing?")
        self._model = step3_finetune(model, self._pert_data, gene_ids)
        self._gene_ids = gene_ids
        adata_pred = step4_predict(self._model, self._pert_data, gene_ids)
        adata_real = ad.read_h5ad(self.config.processed_dir / "norman.h5ad")
        return evaluate_dim_a(adata_pred, adata_real)

    def fine_tune(self) -> object:  # pragma: no cover
        """Just the fine-tuning step (no eval). Useful for cached-prediction tests."""
        try:
            from src.models.run_scgpt_perturbation import (  # noqa: WPS433
                step1_load_data,
                step2_build_vocab_and_model,
                step3_finetune,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "scGPT fine_tune() requires the scgpt + GEARS runtime."
            ) from exc
        self._pert_data = step1_load_data()
        model, vocab, gene_ids = step2_build_vocab_and_model(self._pert_data)
        if model is None:
            raise RuntimeError("scGPT step2 returned None")
        self._model = step3_finetune(model, self._pert_data, gene_ids)
        self._gene_ids = gene_ids
        return self._model
