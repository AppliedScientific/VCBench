"""
Download pySCENIC reference databases for GRN inference baseline.

Downloads:
- CisTarget ranking database (hg38, 10kb upstream/downstream)
- Motif-to-TF annotation table
- Human TF list
"""

import os
import subprocess

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SCENIC_DIR = os.path.join(RAW_DIR, "scenic")

FILES = {
    "rankings": {
        "url": (
            "https://resources.aertslab.org/cistarget/databases/"
            "homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/"
            "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
        ),
        "filename": "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    },
    "motifs": {
        "url": (
            "https://resources.aertslab.org/cistarget/motif2tf/"
            "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
        ),
        "filename": "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
    },
    "tf_list": {
        "url": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
        "filename": "allTFs_hg38.txt",
    },
}


def download():
    os.makedirs(SCENIC_DIR, exist_ok=True)

    for name, info in FILES.items():
        out_path = os.path.join(SCENIC_DIR, info["filename"])
        if os.path.exists(out_path):
            print(f"Already downloaded: {out_path}")
            continue

        print(f"Downloading pySCENIC {name}...")
        subprocess.run(["wget", "-O", out_path, info["url"]], check=True)
        print(f"  -> {out_path}")


def verify():
    for name, info in FILES.items():
        path = os.path.join(SCENIC_DIR, info["filename"])
        assert os.path.exists(path), f"Missing: {path}"
        size_mb = os.path.getsize(path) / 1e6
        print(f"pySCENIC {name}: {size_mb:.1f} MB")

    print("pySCENIC databases verified.")
    return True


if __name__ == "__main__":
    download()
    verify()
