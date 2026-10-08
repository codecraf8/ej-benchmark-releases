# ej benchmark

Code and aggregate results of the benchmark that compares ej with other typed-decision models. Method, rival choice and
contamination audit: [METHOD.md](METHOD.md). Results: [results/](results/).

## What it measures

Every model gets the same records (a `state` plus typed questions, the ej input format, with the gold answers removed)
and must return one probability distribution per question. `scoring.py` scores them per suite:

| metric | meaning |
|---|---|
| NLL | mean negative log-probability of the gold option (lower is better; the headline metric) |
| acc | accuracy of the most probable option |
| ECE15 | expected calibration error over 15 confidence bins |
| CA | certified automation: coverage on one half of the questions of a confidence threshold certified on the other half to keep the error rate <= 10% (Clopper-Pearson, delta 0.10) |
| soft_ce | cross-entropy against gold distributions, where a suite provides them |
| ms/record | wall-clock latency per record, one record per call, after a warm-up record |

## Suites: public sources vs private data

| suite | records / questions | source | public? |
|---|---|---|---|
| td | 300 / 1,500 | Typed Decisions **test** split, the three workflows ej was trained on | public dataset (Apache-2.0); suite files not in this repository |
| zs_td | 100 / 500 | Typed Decisions test split, workflow `security_incidents` (held out of ej's training) | public dataset; suite files not in this repository |
| zs_massive | 1,000 / 1,000 | MASSIVE (en) intent **test** split, options sampled leak-free (every option equally likely to be the gold) | public dataset (CC BY 4.0); suite files not in this repository |
| tickets | 169 / 507 | private support-ticket corpus, test split (same distribution as part of ej's training data) | **private** |
| tickets_ood | 147 / 441 | private support-ticket corpus, out-of-distribution test split | **private** |

The suites are sealed test sets: they are scored once per release and never used for model selection. **None of the
suite files ships with this repository**, so `run_bench.py` cannot reproduce the published numbers from this repository
alone; the code is published as the reference for how they were produced, and it runs unchanged on any suite file in the
ej input format with gold answers (`scoring.py` docstring) — for example your own labelled records.

## Files

| path | what |
|---|---|
| `run_bench.py` | runner: blind records, one record per call, `scoring.check`, summary JSON per model and suite |
| `scoring.py` | NLL, accuracy, ECE15, soft cross-entropy, certified automation |
| `rivals/` | one adapter per model (`rivals/__init__.py` lists them with the environment each needs) |
| `METHOD.md` | rival set, contamination audit, per-suite fairness, adapter rules, methodological choices |
| `results/run2/` | final benchmark of ej 1.0.0: `RUN2.md` table + `*.summary.json` aggregates |
