"""
Download TRRUST v2 human transcription factor-target database.

Source: https://www.grnpedia.org/trrust/
Expected: ~8,444 regulatory interactions
"""

import os
import subprocess

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "trrust_rawdata.human.tsv")
TRRUST_URLS = [
    "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    "https://web.archive.org/web/2024/https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
]


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        print(f"Already downloaded: {OUTPUT_FILE}")
        return OUTPUT_FILE

    for url in TRRUST_URLS:
        print(f"Trying TRRUST download: {url}")
        result = subprocess.run(
            ["wget", "--timeout=30", "--tries=2", "-O", OUTPUT_FILE, url],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.getsize(OUTPUT_FILE) > 1000:
            print(f"Downloaded: {OUTPUT_FILE}")
            return OUTPUT_FILE
        print(f"  Failed (rc={result.returncode}), trying next URL...")

    raise RuntimeError("All TRRUST download URLs failed. Site may be down.")


def verify():
    import pandas as pd

    df = pd.read_csv(
        OUTPUT_FILE,
        sep="\t",
        header=None,
        names=["TF", "Target", "Mode", "PMID"],
    )
    assert len(df) > 8000, f"Expected >8000 rows, got {len(df)}"
    print(f"TRRUST verified: {len(df)} interactions, {df['TF'].nunique()} TFs")
    return True


if __name__ == "__main__":
    download()
    verify()
