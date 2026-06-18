"""VCBench — capability-stratified benchmark for single-cell foundation models.

The three reusable methodological tools from VCBench (2026) are exposed as
top-level modules so they can be imported and used independently of the rest of
the benchmark harness:

* ``vcbench.protocols`` — common-label-set protocol (Eq. 5).
* ``vcbench.probes`` — spread-error correlation probe (Eq. 10).
* ``vcbench.contamination`` — pretraining-overlap manifest schema and validator.
"""

__version__ = "0.1.0"

from vcbench import contamination, probes, protocols  # noqa: F401

__all__ = ["protocols", "probes", "contamination", "__version__"]
