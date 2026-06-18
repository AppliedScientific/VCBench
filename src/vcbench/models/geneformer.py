"""Geneformer V2-316M wrapper.

Regimes: A=FT+D, B=ZS, C=IE, D=ZS+D, E=ZS. Census-independent training corpus
(Genecorpus-30M excludes immortalised / malignant cell lines).

Dim A pipeline (Ahlmann-Eltze et al. 2025 protocol):

1. Prepare data — add ensembl_id + n_counts columns required by the tokeniser.
2. Tokenise to Geneformer rank-value sequences (geneformer.tokenizer).
3. Fine-tune ``BertForSequenceClassification`` on perturbation labels
   (247 classes for Norman, 3 epochs, lr 5e-5, per-device batch 12,
   bottom 2 transformer layers frozen, AdamW, weight decay 0.01,
   warmup 100 steps).
4. Extract control CLS embeddings via ``geneformer.EmbExtractor``.
5. Extract perturbed CLS embeddings via direct ``get_embs()`` call on
   token sequences with the perturbed gene tokens removed
   (``InSilicoPerturber.perturb_data()`` only saves cosine sims, not
   raw embeddings — we bypass it).
6. Train ridge decoder (α=1.0) mapping mean CLS embedding → mean
   expression on training perturbations.
7. Apply decoder to test perturbations + evaluate with the
   :func:`vcbench.dimensions.dim_a_perturbation.evaluate_dim_a`.

Reference: Geneformer V2-316M FT+D PRR on Norman additive-evaluable
71-pert subset = 0.627.

Engineering note:
This wrapper composes verified step functions from
``src.models.run_geneformer_perturbation`` rather than re-implementing
them. The functions there are tested against the on-disk
``results/dim_a/geneformer/cell_eval_results.csv`` artefact (validated by
`tests/unit/test_dim_a_evaluate.py::test_baseline_on_disk_values_match_reference_fixture`).
The wrapper provides the public ABI and state management; the legacy
module owns the GPU runtime details.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vcbench.models.base import FoundationModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData

    from vcbench.dimensions.dim_a_perturbation.evaluate import DimAResult


# Repo-relative defaults; override per-instance if running outside the layout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR: Path = (
    _REPO_ROOT / "models" / "geneformer" / "Geneformer" / "Geneformer-V2-316M"
)
DEFAULT_RESULTS_DIR: Path = _REPO_ROOT / "results" / "dim_a" / "geneformer"
DEFAULT_PROCESSED_DIR: Path = _REPO_ROOT / "data" / "processed"
DEFAULT_TOKENIZED_DIR: Path = _REPO_ROOT / "data" / "tokenized"


@dataclass
class GeneformerConfig:
    """Runtime configuration for the Geneformer wrapper.

    Defaults reproduce the canonical protocol exactly. Override only for tests / ablations.
    """

    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    finetuned_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR / "finetuned_classifier")
    results_dir: Path = DEFAULT_RESULTS_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    tokenized_dir: Path = DEFAULT_TOKENIZED_DIR

    # Dim A fine-tuning hyperparameters
    epochs: int = 3
    learning_rate: float = 5e-5
    per_device_batch_size: int = 12
    n_layers_to_freeze: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Ridge decoder
    decoder_ridge_alpha: float = 1.0


class Geneformer(FoundationModel):
    """Geneformer V2-316M wrapper.

    Constructed cold; call :meth:`load_pretrained` to point at a checkpoint
    directory. The full Dim A pipeline runs via :meth:`run_dim_a` (composes
    the legacy step functions and returns a
    :class:`vcbench.dimensions.dim_a_perturbation.evaluate.DimAResult`).
    """

    name = "Geneformer V2-316M"

    per_dimension_regime = {
        "A": "FT+D",
        "B": "ZS",
        "C": "IE",
        "D": "ZS+D",
        "E": "ZS",
    }

    def __init__(self, config: GeneformerConfig | None = None) -> None:
        self.config = config or GeneformerConfig()
        self.checkpoint_path: str | None = None
        self._finetuned: object | None = None  # path or model object after step3

    # ------------------------------------------------------------------
    # FoundationModel contract

    def load_pretrained(self, checkpoint_path: str | os.PathLike) -> None:
        """Pin the checkpoint path. Heavy load happens in :meth:`run_dim_a`."""
        self.checkpoint_path = str(checkpoint_path)
        self.config.checkpoint_dir = Path(checkpoint_path)

    def embed(self, adata: "AnnData") -> "np.ndarray":  # pragma: no cover
        """Return CLS embeddings for the cells in ``adata``.

        Delegates to ``geneformer.EmbExtractor`` via
        ``src.models.run_geneformer_perturbation.step4_extract_control_embeddings``.
        Requires the ``geneformer`` Python package and a GPU at runtime.
        """
        try:
            from src.models.run_geneformer_perturbation import (  # noqa: WPS433
                step4_extract_control_embeddings,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Geneformer embed() requires the legacy runtime under "
                "src/models/run_geneformer_perturbation.py and the geneformer "
                "Python package. Install via configs/environments/vcbench-pt118."
            ) from exc
        if self._finetuned is None:
            raise RuntimeError(
                "embed() requires a fine-tuned classifier; call run_dim_a() first "
                "or set self._finetuned to a checkpoint directory."
            )
        return step4_extract_control_embeddings(self._finetuned)

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":  # pragma: no cover
        """Predict per-perturbation expression via the fine-tuned classifier + ridge decoder.

        Composes legacy steps 5 (perturbed embedding extraction) + 6 (decoder
        train) + 7 (predict). End-to-end orchestration lives in :meth:`run_dim_a`;
        this method is the lower-level inference path used by integration tests.
        """
        try:
            from src.models.run_geneformer_perturbation import (  # noqa: WPS433
                step5_extract_perturbed_embeddings,
                step6_train_decoder,
                step7_predict_and_evaluate,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Geneformer predict_perturbation() requires the legacy runtime "
                "under src/models/run_geneformer_perturbation.py."
            ) from exc
        if self._finetuned is None:
            raise RuntimeError("call run_dim_a() or load a fine-tuned classifier first")
        ctrl_emb = self.embed(adata=None)  # type: ignore[arg-type]  (legacy signature)
        pert_embeddings = step5_extract_perturbed_embeddings(self._finetuned)
        decoder = step6_train_decoder(ctrl_emb, pert_embeddings)
        return step7_predict_and_evaluate(decoder, pert_embeddings)

    def extract_gene_attention(self) -> "np.ndarray":  # pragma: no cover
        """Per-(TF, target) attention scores from the released checkpoint (regime IE).

        Used only for Dim C; runtime path lives in
        ``src/models/run_grn_baselines.py`` (legacy).
        """
        raise NotImplementedError(
            "Geneformer attention extraction (Dim C IE) is in the legacy pipeline. "
            "The released checkpoint is loaded "
            "with attention output enabled and pooled at layer 13."
        )

    # ------------------------------------------------------------------
    # Dim A orchestration

    def run_dim_a(self) -> "DimAResult":  # pragma: no cover
        """End-to-end Dim A pipeline. Returns a DimAResult.

        Composes legacy steps 1–7 from
        ``src.models.run_geneformer_perturbation`` and re-evaluates the
        predictions through the new
        :func:`vcbench.dimensions.dim_a_perturbation.evaluate_dim_a` so the
        result schema matches the rest of the new package.
        """
        try:
            from src.models.run_geneformer_perturbation import (  # noqa: WPS433
                step1_prepare_data,
                step2_tokenize,
                step3_finetune_classifier,
                step4_extract_control_embeddings,
                step5_extract_perturbed_embeddings,
                step6_train_decoder,
                step7_predict_and_evaluate,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Geneformer run_dim_a() requires the legacy runtime + geneformer + "
                "transformers + torch. Install via configs/environments/vcbench-pt118."
            ) from exc
        import anndata as ad

        from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a

        prepared = step1_prepare_data()
        tokenized = step2_tokenize()
        self._finetuned = step3_finetune_classifier()
        ctrl_emb = step4_extract_control_embeddings(self._finetuned)
        pert_embeddings = step5_extract_perturbed_embeddings(self._finetuned)
        decoder = step6_train_decoder(ctrl_emb, pert_embeddings)
        adata_pred = step7_predict_and_evaluate(decoder, pert_embeddings)

        adata_real = ad.read_h5ad(self.config.processed_dir / "norman.h5ad")
        return evaluate_dim_a(adata_pred, adata_real)
