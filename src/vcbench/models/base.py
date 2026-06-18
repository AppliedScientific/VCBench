"""Abstract foundation-model interface.

Every model wrapper subclasses :class:`FoundationModel` and either implements
the relevant capability methods or raises ``NotImplementedError`` with a
documented reason. This makes the N/A / DNR designations in §I.4
('UCE 33-layer N/A on Dim A — produces embeddings but not expression vectors'
etc.) directly traceable to the implementing class.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData


class FoundationModel(abc.ABC):
    """Abstract base class for every foundation model in the VCBench harness.

    Concrete subclasses MUST implement :meth:`load_pretrained` and
    :meth:`embed`. They MAY override :meth:`predict_perturbation` and
    :meth:`extract_gene_attention` if the underlying model supports those
    capabilities; otherwise the base class raises ``NotImplementedError``
    with a structured ``CapabilityNotSupported`` payload.

    Per-model regime designations from §I.4 are encoded as the
    :attr:`per_dimension_regime` class attribute so the capability matrix
    is queryable without instantiation.
    """

    name: str = ""
    """Display name (e.g. ``'Geneformer V2-316M'``)."""

    per_dimension_regime: dict[str, str] = {}
    """Per-dimension regime label per §I.4 — e.g. ``{'A': 'FT+D', 'B': 'ZS', ...}``.
    Use ``'N/A'`` for structurally-N/A dimensions and ``'DNR'`` for did-not-run."""

    @abc.abstractmethod
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint path."""

    @abc.abstractmethod
    def embed(self, adata: "AnnData") -> "np.ndarray":
        """Return cell embeddings for ``adata.X``, shape ``(n_cells, embedding_dim)``."""

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":
        """Return predicted post-perturbation expression for the given perturbation IDs.

        Subclasses override this iff the model supports perturbation-conditioned
        generation. Default raises ``NotImplementedError`` with a structured
        explanation of why the dimension is N/A or DNR for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement perturbation-conditioned "
            "generation. Per the §I.4 capability matrix this dimension is N/A or DNR."
        )

    def extract_gene_attention(self) -> "np.ndarray":
        """Return per-(TF, target) attention scores for GRN inference.

        Subclasses override this iff the model exposes a gene-level attention
        or similarity surface compatible with the BEELINE / TRRUST evaluation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose gene-level attention. "
            "Per the §I.4 capability matrix this dimension is N/A or DNR."
        )

    # ---------------------------------------------------------------------
    # Convenience accessors

    def regime_for(self, dimension: str) -> str:
        """Return the §I.4 regime label for one dimension (``'A'..'E'``)."""
        return self.per_dimension_regime.get(dimension.upper(), "unspecified")

    def is_dimension_supported(self, dimension: str) -> bool:
        """True iff ``regime_for(dimension)`` is neither N/A nor DNR."""
        regime = self.regime_for(dimension)
        return regime not in ("N/A", "DNR", "unspecified")
