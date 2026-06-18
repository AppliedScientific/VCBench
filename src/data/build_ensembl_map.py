"""
One-time script: generate static Ensembl gene symbol -> ID mapping.

Run once during benchmark setup, then commit the output file.
The static file is used by src/models/data_prep.py for reproducible
gene ID mapping without network dependencies.

Usage:
    python -m src.data.build_ensembl_map
"""

import json
import os

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "static")
OUTPUT_FILE = os.path.join(STATIC_DIR, "ensembl_symbol_map.json")

BIOMART_HOSTS = [
    "http://www.ensembl.org",
    "http://useast.ensembl.org",
    "http://asia.ensembl.org",
]


def build():
    from pybiomart import Dataset as BioMartDataset

    os.makedirs(STATIC_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Already exists: {OUTPUT_FILE}")
        return

    for host in BIOMART_HOSTS:
        try:
            print(f"Querying BioMart ({host})...")
            bm = BioMartDataset(name="hsapiens_gene_ensembl", host=host)
            mapping = bm.query(
                attributes=["external_gene_name", "ensembl_gene_id"]
            )
            symbol_map = dict(
                zip(mapping["Gene name"], mapping["Gene stable ID"])
            )

            with open(OUTPUT_FILE, "w") as f:
                json.dump(symbol_map, f)

            print(f"Saved {len(symbol_map)} mappings -> {OUTPUT_FILE}")
            return
        except Exception as e:
            print(f"  Failed ({host}): {e}")

    raise RuntimeError("All BioMart hosts failed. Retry later.")


if __name__ == "__main__":
    build()
