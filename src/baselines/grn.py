"""
GRN inference baselines for Dimension C.

Baselines:
- Co-expression: Spearman correlation between TFs and all genes
- pySCENIC: GRNBoost2 + motif enrichment (must run in vcbench-scenic env)
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr


def coexpression_baseline(adata, tf_list, top_k=10000):
    """
    Spearman correlation between TF and all genes as GRN proxy.

    Uses vectorized full-matrix correlation for efficiency.
    With ~800 TFs x ~5K genes on 758 cells, this is tractable (~1-5 min).
    """
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    gene_names = list(adata.var_names)
    tf_indices = [i for i, g in enumerate(gene_names) if g in tf_list]

    if not tf_indices:
        raise ValueError("No TFs found in gene list. Check vocabulary overlap.")

    # Compute full Spearman correlation matrix in one call (much faster than loops)
    rho_matrix, _ = spearmanr(X)

    edges = []
    for i in tf_indices:
        for j in range(X.shape[1]):
            if i != j:
                edges.append({
                    "TF": gene_names[i],
                    "target": gene_names[j],
                    "score": abs(rho_matrix[i, j]),
                })

    df = pd.DataFrame(edges)
    return df.nlargest(top_k, "score")


def scenic_baseline(expression_csv_path, tf_file, db_files, motif_file):
    """
    pySCENIC GRN inference wrapper.

    MUST run in vcbench-scenic environment (Python 3.8) due to dependency
    conflicts with scanpy>=1.9 on Python 3.10+.

    Args:
        expression_csv_path: Path to expression matrix CSV (cells x genes)
        tf_file: Path to TF list (allTFs_hg38.txt)
        db_files: List of paths to CisTarget ranking .feather databases
        motif_file: Path to motif-to-TF annotation table
    """
    from arboreto.algo import grnboost2
    from arboreto.utils import load_tf_names
    from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
    from pyscenic.prune import df2regulons, prune2df
    from pyscenic.utils import modules_from_adjacencies

    # Load expression matrix
    expression_matrix = pd.read_csv(expression_csv_path, index_col=0)

    tf_names = load_tf_names(tf_file)

    print("Running GRNBoost2...")
    adjacencies = grnboost2(expression_matrix, tf_names=tf_names, verbose=True)

    print("Pruning with CisTarget motif enrichment...")
    dbs = [
        RankingDatabase(fname=f, name=f.split("/")[-1]) for f in db_files
    ]
    modules = modules_from_adjacencies(adjacencies, expression_matrix)
    df = prune2df(dbs, modules, motif_file)
    regulons = df2regulons(df)

    scenic_edges = []
    for reg in regulons:
        tf = reg.name.rstrip("(+)").rstrip("(-)")
        for target, weight in reg.gene2weight.items():
            scenic_edges.append({"TF": tf, "target": target, "score": weight})

    return pd.DataFrame(scenic_edges)


def degree_null_baseline(predicted_df, gene_names, n_shuffles=100, seed=42):
    """
    Degree-preserving null baseline for GRN evaluation.

    Generates random edge lists that preserve the out-degree distribution
    of each TF from the predicted network. Used as the second passing
    criterion: a model must exceed both pySCENIC AND degree-null AUPRC.

    Args:
        predicted_df: DataFrame with columns ['TF', 'target', 'score']
        gene_names: List of all gene names in the vocabulary
        n_shuffles: Number of random shuffles to average over
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns ['TF', 'target', 'score'] where scores
        are averaged across shuffles (expected: ~uniform)
    """
    rng = np.random.default_rng(seed)

    # Compute out-degree per TF from predictions
    tf_degree = predicted_df.groupby("TF").size().to_dict()
    all_targets = list(set(gene_names))

    # Generate shuffled edge lists preserving TF out-degree
    pair_scores = {}
    pair_counts = {}

    for shuffle_idx in range(n_shuffles):
        for tf, degree in tf_degree.items():
            # Random targets (without replacement if possible)
            n_targets = min(degree, len(all_targets) - 1)
            candidates = [g for g in all_targets if g != tf]
            random_targets = rng.choice(candidates, size=n_targets, replace=False)

            for rank, target in enumerate(random_targets):
                pair = (tf, target)
                # Score = inverse rank (higher rank = higher score)
                score = 1.0 - (rank / n_targets)
                pair_scores[pair] = pair_scores.get(pair, 0.0) + score
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Average scores across shuffles
    edges = []
    for (tf, target), total_score in pair_scores.items():
        edges.append({
            "TF": tf,
            "target": target,
            "score": total_score / pair_counts[(tf, target)],
        })

    df = pd.DataFrame(edges)
    if len(df) > 0:
        df = df.sort_values("score", ascending=False)
    return df
