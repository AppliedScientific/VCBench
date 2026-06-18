"""
Download Norman combinatorial CRISPR dataset via GEARS Python API.

Source: GEARS package (auto-downloads ~1.59 GB)
Contains: >80K cells, >90 single perturbations, >120 double perturbations
"""

import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def download():
    from gears import PertData

    os.makedirs(RAW_DIR, exist_ok=True)

    expected = os.path.join(RAW_DIR, "norman", "perturb_processed.h5ad")
    if os.path.exists(expected):
        print(f"Already downloaded: {expected}")
        return expected

    print("Downloading Norman dataset via GEARS API...")
    pert_data = PertData(RAW_DIR)
    pert_data.load(data_name="norman")
    print(f"Downloaded: {expected}")
    return expected


def verify():
    import anndata as ad

    path = os.path.join(RAW_DIR, "norman", "perturb_processed.h5ad")
    adata = ad.read_h5ad(path)
    assert adata.n_obs > 80_000, f"Expected >80K cells, got {adata.n_obs}"

    conds = adata.obs["condition"].unique()
    singles = [c for c in conds if "+ctrl" in c or "ctrl+" in c]
    doubles = [c for c in conds if "+" in c and "ctrl" not in c]

    assert len(singles) > 90, f"Expected >90 singles, got {len(singles)}"
    assert len(doubles) > 120, f"Expected >120 doubles, got {len(doubles)}"

    print(
        f"Norman verified: {adata.n_obs} cells, "
        f"{len(singles)} singles, {len(doubles)} doubles, "
        f"control: {'ctrl' in conds}"
    )
    return True


if __name__ == "__main__":
    download()
    verify()
