# Worked example — running `vcbench-check-contamination` against Arc State

This demonstrates the schema validator's behavior against a real released
foundation model (Arc State) which has **not yet adopted** the v1 schema.

Run the validator against any current single-cell FM release and you
will get `schema_incomplete` or `unknown`, because **no public model
has yet published a
`training_cells.md5.txt` manifest**. That is precisely the gap the
schema is designed to close.

## Reproducing the demo

```bash
# Step 1: synthesize a minimal release directory mirroring what an Arc
# State v1-compliant release would look like.
mkdir -p /tmp/arc_state_release
echo "abcd1234ef567890,GSE149383,A549_ctrl_001" > /tmp/arc_state_release/training_cells.md5.txt
cat > /tmp/arc_state_release/pretraining_manifest.yaml <<YAML
schema_version: vcbench-contamination-v1
model: arc_state_synthetic_demo
included_datasets: [GSE149383]
excluded_datasets: [GSE133344]
YAML

# Step 2: run against Norman (which Arc State did NOT train on) — expected verdict: unlikely
.venv/bin/python -m tools.vcbench_contamination_check.check \
    /tmp/arc_state_release \
    data/processed/norman.h5ad

# Expected output: "unlikely — accession GSE133344 is in excluded_datasets"
# Exit code: 0
```

## What this demonstrates

The schema produces a definitive verdict in <1 second, replacing
platform-inference audits with direct overlap detection.

### Verdict paths covered by the test suite

| verdict | meaning | exit code | example invocation |
|---|---|---|---|
| `confirmed` | training manifest contains MD5 hashes that match evaluation cells | 1 | TF eval on a cell that appears in the training manifest |
| `likely` | no MD5 match, but eval accession is in `included_datasets` | 1 | scGPT eval on a CELLxGENE Census cell that scGPT was trained on |
| `unlikely` | no MD5 match, eval accession is in `excluded_datasets` | 0 | scGPT eval on Replogle (which scGPT excludes) |
| `unknown` | no MD5 match, eval accession not listed | 0 | scGPT eval on a dataset the manifest doesn't mention |
| `schema_incomplete` | manifest files missing | 2 | any current FM release |

All five paths are covered by `tools/vcbench_contamination_check/tests/test_check.py`
(7 tests, all passing).

## Real-world status

| Model | training_cells.md5.txt published? | pretraining_manifest.yaml? | verdict against any eval |
|---|:---:|:---:|:---|
| Geneformer V2-316M | ❌ | ❌ | `schema_incomplete` |
| scGPT_human | ❌ | ❌ | `schema_incomplete` |
| UCE 33-layer | ❌ | ❌ | `schema_incomplete` |
| TranscriptFormer | ❌ | ❌ | `schema_incomplete` |
| Arc State | ❌ | ❌ | `schema_incomplete` |

This is the motivation for the schema: it asks for **disclosure only**,
no retraining, and the cost is one hash file per cell already in the
training corpus.
