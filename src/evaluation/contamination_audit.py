"""
Contamination audit: Table 3 (contamination matrix).

Cross-references each model's documented training data sources against
our evaluation datasets to flag potential train-test overlap.

Risk levels:
- "confirmed": Dataset explicitly listed in model's training data
- "likely": Dataset present in training corpus (e.g., CELLxGENE Census)
- "unlikely": Dataset excluded by corpus inclusion criteria or independent
- "unknown": Insufficient documentation to determine

Sources: Model papers, GitHub READMEs, HuggingFace manifests,
CELLxGENE Census inclusion/exclusion schema.
"""

import json
import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "tables")

# Contamination risk matrix based on published documentation
# Format: (model, dataset) → {"risk": level, "rationale": explanation}
CONTAMINATION_MATRIX = {
    # --- Geneformer V2-316M ---
    # Trained on Genecorpus-30M (curated from GEO/SRA, excludes cell lines)
    ("Geneformer", "Norman"): {
        "risk": "unlikely",
        "rationale": "Genecorpus-30M excludes cell lines; Norman uses K562 cell line",
    },
    ("Geneformer", "Replogle"): {
        "risk": "unlikely",
        "rationale": "Genecorpus-30M excludes cell lines; Replogle uses K562/RPE1",
    },
    ("Geneformer", "CITE-seq"): {
        "risk": "unlikely",
        "rationale": "Independent corpus (pre-Census); bone marrow may be included but not ADT data",
    },
    ("Geneformer", "Weinreb"): {
        "risk": "unlikely",
        "rationale": "Genecorpus-30M is human-only; Weinreb is mouse HSPCs",
    },
    ("Geneformer", "sci-fate"): {
        "risk": "unlikely",
        "rationale": "A549 cell line excluded from Genecorpus",
    },
    # --- scGPT ---
    # Trained on CELLxGENE Census 2023-05-15 (excludes cell culture, perturbation assays)
    ("scGPT", "Norman"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture and perturbation assays",
    },
    ("scGPT", "Replogle"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture and perturbation assays",
    },
    ("scGPT", "CITE-seq"): {
        "risk": "likely",
        "rationale": "GSE194122 present in Census 2023-05-15; RNA portion may overlap",
    },
    ("scGPT", "Weinreb"): {
        "risk": "unlikely",
        "rationale": "In vitro culture excluded from Census inclusion criteria",
    },
    ("scGPT", "sci-fate"): {
        "risk": "unlikely",
        "rationale": "A549 cell line excluded from Census",
    },
    # --- UCE 33-layer ---
    # Trained on CELLxGENE Census (multi-species)
    ("UCE", "Norman"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture",
    },
    ("UCE", "Replogle"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture",
    },
    ("UCE", "CITE-seq"): {
        "risk": "likely",
        "rationale": "Census-derived training data; GSE194122 likely included",
    },
    ("UCE", "Weinreb"): {
        "risk": "unlikely",
        "rationale": "In vitro culture excluded",
    },
    ("UCE", "sci-fate"): {
        "risk": "unlikely",
        "rationale": "Cell line excluded",
    },
    # --- TranscriptFormer ---
    # Trained on CZI Virtual Cells corpus (Census-derived)
    ("TranscriptFormer", "Norman"): {
        "risk": "unlikely",
        "rationale": "Virtual Cells corpus excludes cell culture",
    },
    ("TranscriptFormer", "Replogle"): {
        "risk": "unlikely",
        "rationale": "Virtual Cells corpus excludes perturbation assays",
    },
    ("TranscriptFormer", "CITE-seq"): {
        "risk": "likely",
        "rationale": "CZI builds both Census and TranscriptFormer; high overlap likelihood",
    },
    ("TranscriptFormer", "Weinreb"): {
        "risk": "unlikely",
        "rationale": "In vitro culture excluded",
    },
    ("TranscriptFormer", "sci-fate"): {
        "risk": "unlikely",
        "rationale": "Cell line excluded",
    },
    # --- Arc State ---
    # SE model: Census Jan 2025; ST model: includes perturbation data
    ("Arc State SE", "Norman"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture",
    },
    ("Arc State SE", "Replogle"): {
        "risk": "unlikely",
        "rationale": "Census excludes cell culture",
    },
    ("Arc State SE", "CITE-seq"): {
        "risk": "likely",
        "rationale": "Census Jan 2025 likely includes GSE194122",
    },
    ("Arc State SE", "Weinreb"): {
        "risk": "unlikely",
        "rationale": "In vitro culture excluded",
    },
    ("Arc State SE", "sci-fate"): {
        "risk": "unlikely",
        "rationale": "Cell line excluded",
    },
    ("Arc State ST", "Replogle"): {
        "risk": "confirmed",
        "rationale": "Replogle K562 explicitly in Arc State ST training data list",
    },
    ("Arc State ST", "Norman"): {
        "risk": "unlikely",
        "rationale": "Not listed in Arc State ST training data",
    },
    # The Arc State ST model is Dim A / perturbation-response only; it does
    # not participate in cross-species, GRN, cross-modal, or temporal
    # evaluation, so contamination against those datasets is not applicable.
    ("Arc State ST", "CITE-seq"): {
        "risk": "N/A",
        "rationale": "ST model does not evaluate on Dim D (cross-modal)",
    },
    ("Arc State ST", "Weinreb"): {
        "risk": "N/A",
        "rationale": "ST model does not evaluate on Dim E (temporal)",
    },
    ("Arc State ST", "sci-fate"): {
        "risk": "N/A",
        "rationale": "ST model does not evaluate on Dim E (temporal)",
    },
}


def generate_table():
    """Generate contamination audit table as CSV."""
    import pandas as pd

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = ["Geneformer", "scGPT", "UCE", "TranscriptFormer",
              "Arc State SE", "Arc State ST"]
    datasets = ["Norman", "Replogle", "CITE-seq", "Weinreb", "sci-fate"]

    # Build DataFrame
    rows = []
    for model in models:
        row = {"Model": model}
        for dataset in datasets:
            key = (model, dataset)
            if key in CONTAMINATION_MATRIX:
                entry = CONTAMINATION_MATRIX[key]
                row[dataset] = entry["risk"]
            else:
                row[dataset] = "unknown"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Canonical paper-facing name. The older ``table3_contamination.csv``
    # alias was dropped because it caused divergence between sources of
    # truth.
    ed_csv_path = os.path.join(OUTPUT_DIR, "ed_table1_contamination.csv")
    df.to_csv(ed_csv_path)
    print(f"ED Table 1 saved: {ed_csv_path}")
    print(df.to_string())

    # Also save full rationale as JSON
    json_path = os.path.join(OUTPUT_DIR, "contamination_rationale.json")
    serializable = {
        f"{k[0]}|{k[1]}": v for k, v in CONTAMINATION_MATRIX.items()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Rationale saved: {json_path}")

    # Summary counts
    risks = [v["risk"] for v in CONTAMINATION_MATRIX.values()]
    for level in ["confirmed", "likely", "unlikely", "unknown"]:
        count = risks.count(level)
        print(f"  {level}: {count}")

    return df


def verify_census_inclusion(geo_accession="GSE194122"):
    """
    Optional: Query CELLxGENE Census to verify if a GEO accession is present.

    Requires cellxgene-census installed and internet access.
    """
    try:
        import cellxgene_census

        census = cellxgene_census.open_soma(census_version="stable")
        datasets = (
            census["census_info"]["datasets"].read().concat().to_pandas()
        )
        matches = datasets[
            datasets["dataset_title"].str.contains(
                geo_accession, case=False, na=False
            )
            | datasets["collection_name"].str.contains(
                "NeurIPS|bone marrow", case=False, na=False
            )
        ]
        census.close()

        if len(matches) > 0:
            print(f"\n  Census matches for '{geo_accession}':")
            for _, row in matches.iterrows():
                print(f"    - {row['dataset_title']} ({row['collection_name']})")
        else:
            print(f"\n  No Census matches found for '{geo_accession}'")
        return matches
    except ImportError:
        print("  cellxgene_census not installed; skipping Census verification")
        return None


if __name__ == "__main__":
    generate_table()
