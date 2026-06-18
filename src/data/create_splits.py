"""
Create reproducible train/test split definitions for perturbation prediction.

Two split strategies:
1. GEARS standard split: simulation split with seed=1 (exports perturbation names)
2. Ahlmann-Eltze split: 5 random 50/50 splits of double perturbations
   (Ahlmann-Eltze et al., 2025 protocol)
"""

import os
import pickle

import anndata as ad
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")


def create_gears_split():
    """Export GEARS standard simulation split as model-agnostic .npy files."""
    from gears import PertData

    print("Creating GEARS standard split...")
    pert_data = PertData(RAW_DIR)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=1)

    # Load the split pickle created by GEARS
    split_path = os.path.join(
        RAW_DIR, "norman", "splits", "norman_simulation_1_0.75.pkl"
    )
    with open(split_path, "rb") as f:
        split = pickle.load(f)

    # Export each partition as .npy for model-agnostic consumption
    for partition_name, pert_list in split.items():
        out_path = os.path.join(SPLITS_DIR, f"gears_{partition_name}_perts.npy")
        np.save(out_path, np.array(pert_list))
        print(f"  GEARS {partition_name}: {len(pert_list)} perturbations -> {out_path}")


def create_ahlmann_eltze_splits(n_partitions=5, seed_base=42):
    """
    Ahlmann-Eltze protocol: 5 random 50/50 splits of double perturbations.

    Train = all singles + control + half of doubles
    Test = other half of doubles
    Repeated with 5 different random seeds for mean ± std reporting.
    """
    print(f"Creating Ahlmann-Eltze splits ({n_partitions} partitions)...")
    adata = ad.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))

    ctrl_mask = adata.obs["condition"] == "ctrl"
    single_mask = adata.obs["condition"].str.contains(r"\+ctrl|ctrl\+")

    # Get unique double perturbation names (not cell indices)
    double_perts = sorted([
        c for c in adata.obs["condition"].unique()
        if "+" in c and "ctrl" not in c
    ])
    print(f"  {len(double_perts)} double perturbations found")

    for i in range(n_partitions):
        rng = np.random.default_rng(seed_base + i)
        shuffled = rng.permutation(double_perts)
        n_test = len(shuffled) // 2
        test_perts = set(shuffled[:n_test])
        train_perts = set(shuffled[n_test:])

        # Train = ctrl + singles + train_doubles (cell indices)
        train_idx = np.where(
            ctrl_mask | single_mask |
            adata.obs["condition"].isin(train_perts)
        )[0]
        # Test = test_doubles only
        test_idx = np.where(
            adata.obs["condition"].isin(test_perts)
        )[0]

        assert len(set(train_idx) & set(test_idx)) == 0, (
            f"Train/test overlap in partition {i}!"
        )

        np.save(os.path.join(SPLITS_DIR, f"ae_train_idx_p{i}.npy"), train_idx)
        np.save(os.path.join(SPLITS_DIR, f"ae_test_idx_p{i}.npy"), test_idx)
        np.save(os.path.join(SPLITS_DIR, f"ae_test_perts_p{i}.npy"),
                np.array(sorted(test_perts)))
        print(f"  Partition {i}: Train={len(train_idx)} cells, "
              f"Test={len(test_idx)} cells, "
              f"Test perts={len(test_perts)}")


def run():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    create_gears_split()
    create_ahlmann_eltze_splits()
    print("All splits created.")


if __name__ == "__main__":
    run()
