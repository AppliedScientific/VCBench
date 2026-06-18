"""TranscriptFormer (CZI multi-species, 110M, 12 species) wrapper.

Per §I.4: A=ZS+D, B=ZS (lung+liver only — TF holds species embeddings only
for these two tissues), C=N/A, D=ZS+D, E=ZS+D.

Dim C N/A justification (§I.2): public release lacks gene-attention
extraction interface compatible with BEELINE/TRRUST pipeline. Documented
in Supp Note 2 §S2.1.

Inference config: built in-process via OmegaConf with
``pretrained_embedding=True`` and ``output_keys=["embeddings"]``. No on-disk
YAML used. Perturbation simulation: gene zeroing. Decoder: ridge α=1.0,
fit on (control + training) embeddings, output clipped to [0, GT_max].

Weinreb special case (§I.2): full-dataset DPT eigensolution fails (ARPACK
non-convergence due to near-duplicate embeddings producing degenerate graph
Laplacian). Reported as 10×5K bootstrap subsamples; mean ± std = 0.041 ± 0.078.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vcbench.models.base import FoundationModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData


class TranscriptFormer(FoundationModel):
    name = "TranscriptFormer"
    per_dimension_regime = {
        "A": "ZS+D",
        "B": "ZS",
        "C": "N/A",
        "D": "ZS+D",
        "E": "ZS+D",
    }

    def __init__(self) -> None:
        self.checkpoint_path: str | None = None

    def load_pretrained(self, checkpoint_path: str) -> None:
        self.checkpoint_path = str(checkpoint_path)

    def embed(self, adata: "AnnData") -> "np.ndarray":  # pragma: no cover
        raise NotImplementedError(
            "TranscriptFormer embed() is not exposed through this wrapper. "
            "The OmegaConf-built inference runtime lives in the pipeline at "
            "src/models/run_transcriptformer_perturbation.py and is verified "
            "against the on-disk results/{dim_b,dim_d,dim_e}/transcriptformer/ "
            "reference outputs. The wrapper exposes the ABI surface and the "
            "§I.4 regime declaration."
        )

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":  # pragma: no cover
        raise NotImplementedError(
            "TranscriptFormer predict_perturbation() is not exposed through "
            "this wrapper. It uses gene-zeroing + ridge decoder (α=1.0, output "
            "clipped to [0, GT_max]) per §I.2; the runtime is in the pipeline "
            "at src/models/run_transcriptformer_perturbation.py."
        )

    def extract_gene_attention(self) -> "np.ndarray":
        raise NotImplementedError(
            "TranscriptFormer Dim C is N/A (per Supp Note 2 §S2.1): the "
            "public TF release lacks the gene-attention extraction interface "
            "compatible with BEELINE/TRRUST. This is a structural N/A — "
            "extract_gene_attention is not in scope for any future TF "
            "wrapper version unless CZI ships an attention API."
        )
