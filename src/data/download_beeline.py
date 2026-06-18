"""
Download BEELINE GRN ground truth from Zenodo.

Source: Zenodo record 7682713 (BEELINE v2)
Contains: Ground truth regulatory networks for hESC and hHEP cell types.
"""

import os
import subprocess

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
BEELINE_DIR = os.path.join(RAW_DIR, "beeline")
ZENODO_URL = "https://zenodo.org/api/records/19009603/files/BEELINE-data.zip/content"


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "BEELINE-data.zip")

    if os.path.exists(BEELINE_DIR) and os.path.isdir(BEELINE_DIR):
        print(f"Already downloaded: {BEELINE_DIR}")
        return BEELINE_DIR

    print("Downloading BEELINE ground truth from Zenodo...")
    subprocess.run(["wget", "-O", zip_path, ZENODO_URL], check=True)
    subprocess.run(["unzip", "-o", zip_path, "-d", BEELINE_DIR], check=True)
    os.remove(zip_path)
    print(f"Extracted to: {BEELINE_DIR}")
    return BEELINE_DIR


def verify():
    """Verify that expression data is present for hESC and hHEP.

    The Zenodo archive layout changed between records; expression CSVs
    may live under ``BEELINE-data/inputs/scRNA-Seq/<dataset>/`` (record
    19009603) or ``inputs/Experimental/<dataset>/`` (older records).
    Ground-truth refNetwork.csv is NOT shipped in record 19009603 — the
    TRRUST ground truth (downloaded separately) is the primary evaluation
    target. We therefore verify expression data only.
    """
    for dataset in ["hESC", "hHep"]:
        # Try both known layouts.
        candidates = [
            os.path.join(BEELINE_DIR, "BEELINE-data", "inputs", "scRNA-Seq", dataset, "ExpressionData.csv"),
            os.path.join(BEELINE_DIR, "inputs", "Experimental", dataset, "ExpressionData.csv"),
            os.path.join(BEELINE_DIR, "inputs", "scRNA-Seq", dataset, "ExpressionData.csv"),
        ]
        found = [p for p in candidates if os.path.exists(p)]
        assert found, (
            f"BEELINE {dataset}: ExpressionData.csv not found in any of:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )
        import pandas as pd
        df = pd.read_csv(found[0], index_col=0, nrows=5)
        print(f"BEELINE {dataset}: ExpressionData.csv OK ({df.shape[1]} cells sample) at {found[0]}")

    print("BEELINE expression data verified.")
    return True


if __name__ == "__main__":
    download()
    verify()
