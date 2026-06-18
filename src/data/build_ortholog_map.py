"""
Build human-mouse one-to-one ortholog mapping via Ensembl BioMart.

Outputs a pickle with bidirectional mapping dicts (h2m, m2h).
Expected: ~15,000-16,000 high-confidence one-to-one orthologs.

Reproducibility note: Ensembl updates change mappings over time. Run once during
initial benchmark construction, then host the resulting ortholog_maps.pkl as a
versioned artifact (Zenodo/HuggingFace/GitHub releases). Future users should
download the frozen mapping rather than re-querying.
"""

import os
import pickle

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "ortholog_maps.pkl")

# Ensembl BioMart hosts — cycle through on timeout
BIOMART_HOSTS = [
    "http://www.ensembl.org",
    "http://useast.ensembl.org",
    "http://asia.ensembl.org",
]


def build_ortholog_map():
    from pybiomart import Dataset

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Already built: {OUTPUT_FILE}")
        return

    for host in BIOMART_HOSTS:
        try:
            print(f"Querying Ensembl BioMart ({host})...")
            dataset = Dataset(name="hsapiens_gene_ensembl", host=host)

            orthologs = dataset.query(
                attributes=[
                    "ensembl_gene_id",
                    "external_gene_name",
                    "mmusculus_homolog_ensembl_gene",
                    "mmusculus_homolog_associated_gene_name",
                    "mmusculus_homolog_orthology_type",
                    "mmusculus_homolog_orthology_confidence",
                ]
            )

            # Filter to high-confidence one-to-one orthologs
            one2one = orthologs[
                (orthologs["Mouse orthology type"] == "ortholog_one2one")
                & (orthologs["Mouse orthology confidence [0 low, 1 high]"] == 1)
            ].dropna(subset=["Mouse gene name"])

            human_to_mouse = dict(
                zip(one2one["Gene name"], one2one["Mouse gene name"])
            )
            mouse_to_human = dict(
                zip(one2one["Mouse gene name"], one2one["Gene name"])
            )

            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump({"h2m": human_to_mouse, "m2h": mouse_to_human}, f)

            # Record Ensembl version
            version = getattr(dataset, "display_name", "unknown")
            print(f"Ensembl version: {version}")
            print(f"One-to-one orthologs: {len(one2one)}")
            print(f"Saved: {OUTPUT_FILE}")
            return

        except Exception as e:
            print(f"  Failed ({host}): {e}")
            continue

    raise RuntimeError("All Ensembl BioMart hosts failed. Retry later.")


if __name__ == "__main__":
    build_ortholog_map()
