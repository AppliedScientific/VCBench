# Contributing to VCBench

VCBench is built so that future model developers can submit a new foundation model and have it scored against the same pre-registered baselines as the v1 models. This file is the binding contract for that workflow.

## Two ways to contribute

1. **Submit a new model** — most common. PR adds a wrapper + a manifest + predictions; maintainers verify reproducibility and merge.
2. **Improve the benchmark itself** — bug fix in metric implementation, new baseline, new dimension, doc improvements.

If you're not sure which one applies, open a Discussion first.

## Submitting a new model

Follow the worked template at [`submissions/TEMPLATE.md`](submissions/TEMPLATE.md). At a minimum you provide:

1. **A wrapper class** under `src/vcbench/models/<your_model>.py` that subclasses `vcbench.models.FoundationModel`. Implement `load_pretrained` + `embed`; override `predict_perturbation` and `extract_gene_attention` only if your model supports them. Set `per_dimension_regime` per the §I.4 conventions (`FT`, `FT+D`, `ZS`, `ZS+D`, `IE`, `N/A`, `DNR`).
2. **A contamination manifest** at `src/vcbench/contamination/examples/<your_model>.yaml` declaring the pretraining corpus + included / excluded accessions per the v1 schema. `validate_manifest()` must accept it.
3. **Predictions JSONs** for each dimension you're targeting, in the schema produced by `vcbench.dimensions.dim_X.evaluate_dim_X(...).to_dict()` (or `.to_aggregate_dict()` for Dim A).
4. **Compute environment** — either point at one of the existing `configs/environments/` files or contribute a new one if your model needs custom dependency pins.
5. **A submission memo** at `submissions/<YYYY-MM>-<your_model>.md` filled out from the template.

## Reproducibility expectations

Every numeric claim in your submission memo must be reproducible from the artefacts you provide. Maintainers will:

- Re-run your wrapper end-to-end on a clean machine (CPU for baselines, GPU per your environment for fine-tuning).
- Verify your predictions JSONs match what your wrapper produces, within ±0.001 of the metric values.
- Validate your contamination manifest against `validate_manifest()` and against any independent evidence we have (paper text, training-corpus enumerations).
- Cross-check that your wrapper's `per_dimension_regime` declaration matches what the wrapper actually supports.

Submissions that don't reproduce within tolerance get one round of feedback; if the second submission still doesn't reproduce we close the PR.

## Code quality bar

For wrapper PRs:

- The wrapper file must be readable end-to-end in under 5 minutes (this is the file reviewers from your model's home lab will read first — see §II.18 of the original spec).
- Type annotations on every public function.
- Numpy-style docstrings; the module docstring must include the §I.4 reference values for your model.
- Tests under `tests/unit/test_models_<your_model>.py` covering the wrapper's contract (regime declarations, capability raisers, any train/test-overlap or contamination guard you add).
- All pre-existing tests still pass: `pytest tests/ -q`.

For metric / baseline / dimension PRs:

- Reference values from §I.4 / §I.3 must be locked in `tests/reference_values.json` and validated by drift detectors in `tests/unit/`.
- The relevant `vcbench.dimensions.dim_X.metrics` docstring must cite the manuscript equation number and reference values.
- Hyperparameter changes require a release-notes entry documenting the change and its rationale.

## Pre-registration is read-only

`configs/pre_registration.yaml` is frozen at v1.0. Any change to it requires:

1. A release-notes entry documenting the change and its rationale.
2. A separate PR (not bundled with model submissions).
3. Bump to a new minor version (the file's amendment_history field gets a new entry).

Do not modify `expected_assignments` to make your submission pass — maintainers will catch this and reject the PR.

## Workflow

1. Fork, branch from `main`.
2. Run `pytest tests/ -q` locally before pushing.
3. Open a PR against `main`. CI will fire `test.yml`, `overlap_check.yml`, `baselines.yml`, `docker.yml`.
4. A maintainer assigns review. Expect feedback within 5 working days for code PRs, 14 days for model submissions (longer because we re-run reproductions).
5. Approval requires at least one maintainer review + all CI green.

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Harassment, discrimination, or bad-faith engagement get warned once and then banned.

## Maintainer contacts

- VCBench contributors — open an Issue for bugs, a Discussion for design questions
