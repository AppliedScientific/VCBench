"""Per-model wrappers for the five foundation models evaluated in VCBench v1.

All wrappers implement (or explicitly raise NotImplementedError on) the
:class:`vcbench.models.base.FoundationModel` abstract interface, which is
the binding contract that makes the §I.4 capability matrix's N/A and DNR
designations directly auditable in code.

Per the spec (§II.18), the
:class:`vcbench.models.arc_state.ArcState` wrapper validates its train/test
split in code: its default config points at the Norman GEARS-split TOML, and
a CI test verifies zero overlap between train and test perturbation IDs.
"""

from vcbench.models.arc_state import ArcState
from vcbench.models.base import FoundationModel
from vcbench.models.geneformer import Geneformer
from vcbench.models.scgpt import SCGPT
from vcbench.models.transcriptformer import TranscriptFormer
from vcbench.models.uce import UCE

__all__ = [
    "FoundationModel",
    "Geneformer",
    "SCGPT",
    "UCE",
    "TranscriptFormer",
    "ArcState",
]
