"""
Download cross-species single-cell data from CZI CELLxGENE Census.

Downloads 50K cells per tissue per species for 5 tissues x 2 species = 10 files.
Uses the "stable" LTS release (CZI guarantees 5+ year accessibility).
"""

import os

import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

TISSUES = ["lung", "liver", "heart", "kidney", "brain"]
N_CELLS = 50_000
CENSUS_VERSION = "stable"


def download():
    import cellxgene_census

    os.makedirs(RAW_DIR, exist_ok=True)

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue in TISSUES:
            for org_label, org_key in [
                ("Homo sapiens", "homo_sapiens"),
                ("Mus musculus", "mus_musculus"),
            ]:
                out_path = os.path.join(RAW_DIR, f"census_{org_key}_{tissue}.h5ad")
                if os.path.exists(out_path):
                    print(f"Already downloaded: {out_path}")
                    continue

                print(f"Downloading {org_label} {tissue}...")
                obs = cellxgene_census.get_obs(
                    census,
                    org_key,
                    value_filter=(
                        f"tissue_general == '{tissue}' "
                        f"and is_primary_data == True"
                    ),
                    column_names=[
                        "soma_joinid",
                        "cell_type",
                        "cell_type_ontology_term_id",
                    ],
                )

                if len(obs) > N_CELLS:
                    sampled_ids = np.random.choice(
                        obs["soma_joinid"].values, N_CELLS, replace=False
                    )
                else:
                    sampled_ids = obs["soma_joinid"].values

                adata = cellxgene_census.get_anndata(
                    census,
                    organism=org_label,
                    obs_coords=sampled_ids,
                    obs_column_names=[
                        "cell_type",
                        "cell_type_ontology_term_id",
                        "tissue",
                        "disease",
                        "sex",
                    ],
                    var_column_names=[
                        "feature_id",
                        "feature_name",
                    ],
                )
                adata.write_h5ad(out_path)
                print(f"  {org_label} {tissue}: {adata.shape} -> {out_path}")


def verify():
    import scanpy as sc

    count = 0
    for tissue in TISSUES:
        for org_key in ["homo_sapiens", "mus_musculus"]:
            path = os.path.join(RAW_DIR, f"census_{org_key}_{tissue}.h5ad")
            assert os.path.exists(path), f"Missing: {path}"
            adata = sc.read_h5ad(path)
            assert "cell_type" in adata.obs.columns, f"No cell_type in {path}"
            print(f"  {org_key} {tissue}: {adata.shape}")
            count += 1

    assert count == 10, f"Expected 10 files, found {count}"
    print(f"CELLxGENE Census verified: {count} files")
    return True


if __name__ == "__main__":
    download()
    verify()
