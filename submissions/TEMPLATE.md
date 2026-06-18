# Submission template — `<your-model-name>`

> Copy this file to `submissions/<YYYY-MM>-<your-model>.md` and fill it in.
> Open a PR with this file plus the wrapper code, manifest, predictions, and tests
> per CONTRIBUTING.md.

## Model identity

| Field | Value |
|---|---|
| Model name | `<e.g. CellFlow v1.2>` |
| Authors / lab | `<list>` |
| Source code | `<URL>` |
| Pretrained checkpoint | `<HF / Zenodo / institution URL>` |
| Architecture summary | `<2 sentences max>` |
| Parameter count | `<e.g. 220M>` |
| Pretraining corpus | `<name + citation>` |
| Submission contact | `<name + email>` |

## Submitted predictions

For each dimension you target, fill in the row below and include the predictions JSON path.

| Dim | Submitted? | Regime | Headline metric | Path |
|---|---|---|---|---|
| A: Perturbation | yes / no | `<FT / FT+D / ZS+D / ...>` | PRR=`<value>` | `submissions/predictions/<your-model>/dim_a.json` |
| B: Cross-species | | | macro F1=`<value>` | |
| C: GRN | | | AUPRC + EPR | |
| D: Cross-modal | | | per-protein Pearson=`<value>` | |
| E: Temporal | | | Kendall τ-b=`<value>` | |

## VC Level claim

State which VC Level your model achieves and which dimension(s) drive it. Do not modify `configs/pre_registration.yaml` — maintainers will recompute the level from your submitted numbers.

```
Claimed level: <0 / 1 / 2 / 3>
Binding dimension(s): <e.g. Dim D — your-model 0.31 > mean-celltype 0.152>
```

## Wrapper code

Filed at: `src/vcbench/models/<your_model>.py`

The wrapper must:
- [ ] Subclass `vcbench.models.FoundationModel`
- [ ] Set `name` and `per_dimension_regime` per §I.4 conventions
- [ ] Implement `load_pretrained` and `embed`
- [ ] Override `predict_perturbation` / `extract_gene_attention` only if supported
- [ ] Raise `NotImplementedError` with the manuscript-N/A justification on capabilities the model lacks
- [ ] Pass the regime-declaration and capability-raiser tests at `tests/unit/test_model_wrappers.py` (parametrise your class into the existing fixtures)

## Contamination manifest

Filed at: `src/vcbench/contamination/examples/<your_model>.yaml`

The manifest must:
- [ ] Set `schema_version: vcbench-contamination-v1`
- [ ] List every accession known to be in your training corpus under `included_datasets`
- [ ] List every accession structurally excluded under `excluded_datasets` and `excluded_tissue_types`
- [ ] Validate clean against `validate_manifest()` (parametrise into the existing example-manifest test fixture)
- [ ] Disclose any (model, evaluation-dataset) pair where `included_datasets` overlaps a VCBench evaluation accession (Norman GSE133344, Replogle 10.25452/figshare.plus.20029387, NeurIPS GSE194122, Weinreb GSE140802, sci-fate GSE131351, BEELINE Pratapa 2020)

If your model has a known contamination on any VCBench evaluation accession, the dimension affected must either be excluded from your submission or accompanied by an explicit "we report this anyway because…" rationale in this memo.

## Compute environment

| Step | Hardware | Wall clock | Environment file |
|---|---|---|---|
| Embedding extraction | `<e.g. A100 40GB>` | `<e.g. 2h>` | `<configs/environments/...>` |
| Fine-tuning | | | |
| Inference | | | |

If you contributed a new environment file, link it here.

## Reproducibility statement

```
The submitted predictions JSONs are produced by:

  vcbench predict --model <your-model> --dim a
  vcbench predict --model <your-model> --dim b
  ...

with the wrapper at <commit SHA> on the checkpoint at <checkpoint URL revision>.
The submission was reproduced end-to-end on <hardware> in <wall-clock>
on <date>. Maintainers can reproduce by:

  pip install -e .
  python -m vcbench predict --model <your-model> --dim a
```

## Caveats / known limitations

`<List anything reviewers should know — design-scope mismatches, partial coverage,
known failure modes, deviations from default hyperparameters, etc. Be candid.>`

## Code review preflight

Before requesting review:

- [ ] `pytest tests/ -q` passes locally
- [ ] All CI workflows green on your branch
- [ ] CONTRIBUTING.md reviewed; this template fully filled in
- [ ] `docs/LEADERBOARD.md` updated with a new row for your submission
- [ ] You have read the §II.18 reviewer notes for your model class (in the original spec)
