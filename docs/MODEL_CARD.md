# ej model card (v1.1.0 pending; v1.0.0 withdrawn)

> **The weights are not yet public.** v1.0.0 (state `14a3e64f...`) was **withdrawn before publication**: its training
> pool contained 3,000 Amazon counterfactual records, whose upstream licence is CC BY-NC 4.0, incompatible with the CC
> BY-SA 4.0 weights licence. v1.1.0 is fitted on the licence-clean pool v2b and is pending its pre-registered evaluation
> (seed-aware keep rule with a zs_wide guard, decided before any fit). `V110_*` placeholder fields are filled from that run.

| Field | Value |
|---|---|
| Model | ej 1.1.0 ({{V110_RELEASE_DATE}}); release decision: {{V110_RELEASE_DECISION}} |
| State key | `{{V110_STATE_KEY}}` (recorded in `config.json`; recomputed from `research_commit` + pool by `scripts/state_key.py`) |
| Code | runtime: this repository at `{{V110_CODE_COMMIT}}`; research tree: `{{V110_RESEARCH_COMMIT}}` |
| Size | counted {{V110_SIZE_COUNTED_MIB}} MiB (bit-level bound: encoder codes + vocabulary + int8 heads; no file is stored that way); on disk {{V110_SIZE_ONDISK_MB}} MB (weights directory); download {{V110_SIZE_DOWNLOAD_MB}} MB (weights + the base-model files a first load fetches); resident {{V110_SIZE_RESIDENT_MB}} MB (tensors held while predicting) |
| Latency (CPU, Python reference, 1 thread, one record per call) | {{V110_LATENCY}} (protocol and box: README "Latency") |
| Language | English only (§6) |
| Licence | weights **CC BY-SA 4.0**; code Apache-2.0 |
| Base model | `intfloat/e5-small-v2` (MIT), revision `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| Weights / code | not yet public (Hugging Face `5ak3t/ej` is private) / https://github.com/codecraf8/ej-benchmark-releases |

## 1. What the model does

A **System-1 decision model for devices**: given a `state` (text, or a JSON object as text) and typed questions, it returns
one probability distribution per question in **one pass of a small encoder plus small heads**, without decoding tokens.
Usage: the repository README.

| Question type | Input | Output |
|---|---|---|
| `choice` | instructions + a runtime list of option texts | a probability per option |
| `noul` | a yes/no statement | P(false), P(true) |
| `score` | instructions + an ordered list of level texts | a distribution over the levels |

- **No LLM at inference**; teachers are used at fit time only and their weights do not ship.
- **Calibration is fitted, then checked per suite**: a log-linear pool with per-cell calibration and a lapse term; whether
  the result is calibrated is measured on each suite against a perfect-calibration ECE floor (§5).
- **Record independence.** Zero-shot `predict` is record-independent: each record's output depends only on that record (up
  to the float noise of batching). `adapt` and `predict(..., observe=True)` are an opt-in, per-workflow transductive mode:
  they pool option statistics across that workflow's records, so use them only for one workflow with a fixed option set
  (the same question ids and option keys).
- Option texts are read as text, so new label spaces can be asked zero-shot; quality on unseen workflows is limited (§6).

## 2. Architecture (v1.1.0)

**2.1 Low-bit shared encoder (the only transformer on the device).** Base `intfloat/e5-small-v2`, mean pooling,
`query:` / `passage:` prefixes. Trimmed vocabulary chosen a priori (a dropped piece is re-split, never `[UNK]`). **2-bit**
embeddings and feed-forward matrices, **3-bit attention**, groups of 128 input columns with fp16 step and offset; GPTQ
initialisation, then quantisation-aware distillation to a 4-bit e5. Encoder checkpoint: {{V110_ENCODER}} (fixed and
recorded before fitting). The state is encoded **once** per record; JSON fields are read by key-aware attention.

**2.2 Expert pool, int8 heads.** 9 expert slots in v1.1.0 (v1.0.0: 11): a deep convex conditional logit over
label-agnostic features (zero-shot capable), wide sparse state-token × option-slot crosses, a rich bilinear/ordinal/MLP
scorer, a centred prior-free expert, field attention, relational evidence over JSON cross-field relations, a slot-free
relational reader, a distilled decision-encoder scorer, and a product-of-experts pair (§2.3). The v1.0.0 distilled NLI
pair head is removed: it was exactly 0 on every JSON state and had no measured development benefit. Every head is fitted
on the low-bit encoder's features and compacted to **int8** (per-row int8 matrices; int8 cross weights with 40-bit hashed
keys). Experts that read seen option slots only (attention, relational, slot-free reader) are skipped on batches without
a seen row (identical output).

**2.3 Option-text bias expert.** A **bias-only expert** reads the question and the option texts but not the state; the main
deep expert is trained as a product of experts with the frozen bias logits as an offset (Clark et al. 2019; He et al.
2019), and the bias expert enters the pool with **one signed weight**. On unseen option slots that weight is constrained
to (−1, 0), so the option-text prior is divided out at most once. Whether this helps on a new workflow is a per-suite
question: it is reported against state-free baselines (always the option with the lowest / highest bias logit),
{{V110_POE_BASELINES}}; no transfer claim is made where the model does not beat both.

**2.4 Teachers (fit time only; nothing ships).** A decision encoder (e5 layers 10-12 fine-tuned on the training pool)
distilled from its out-of-fold distributions. No large teacher is used. (v1.0.0 also distilled the NLI cross-encoder
`cross-encoder/nli-deberta-v3-xsmall`; v1.1.0 does not ship that head.)

**2.5 Group-honest stacking.**
- Log-linear pool with lapse per calibration cell (question type × seen/unseen option slots × structured state):
  `p = (1 - eps) softmax(sum_e a_e z_e) + eps / K`; a selective correctness head is adopted per regime only where it beats
  the pool.
- **Group-honest nested cross-fitting**: rows for a held-out group come only from models and teachers that never saw it.
- **New-group predictive, pre-registered.** For each regime of the unseen-slot pool the shipped predictive is the
  plug-in per-cell fit ('flat', no hierarchy) unless the regime has at least 12 training groups and both type-II-ML scales
  lie strictly inside their grid; pool v2b has 7 groups (4 plain-text, 3 JSON), so both regimes ship 'flat'. (v1.0.0
  chose its JSON predictive among four by a .0014-nat leave-one-group-out margin on 3 groups; its hierarchy scales sat at
  the grid edge.)

## 3. Training data (pool v2b)

One fit on the training pool **v2b** (9,719 records, 7 source groups; sha256
`ce1c1a6fecfd62a90317f6efc4f90fd5f9261becb7081fd00235a6f4d5ee9dbe`). Details and counts: `docs/DATA_CARD.md`.

| Source (groups) | Records | Licence |
|---|---|---|
| Typed Decisions, workflows `agent_trace_observability`, `customer_service`, `invoice_processing` (3) | 736 | Apache-2.0 |
| **Support tickets written by Claude Haiku** for this project, train + calibration splits (1) | 780 | in-house; **not released** |
| Banking77 / CLINC150 `plus` / GoEmotions (3) | 3,000 / 3,000 / 2,203 | CC BY 4.0 / CC BY 3.0 / Apache-2.0 |
| **Total** | 9,719 records, 7 groups | |

v1.0.0 used pool v2 (12,719 records, 8 groups): the same plus 3,000 Amazon counterfactual records (CC BY-NC 4.0 upstream,
although the mirror it was taken from declared CC BY 4.0). That is why v1.0.0 was withdrawn.

**Disclosure: Claude Haiku.** The in-domain support tickets were written by Claude Haiku (labels by construction from the
generation spec, no human review). They are part of training; the corpus itself is **not released**, so the training pool
cannot be rebuilt from public data alone. Open licence questions on this and other inputs: `NOTICE` (Q1-Q4).
Not in the training data: Typed Decisions `security_incidents` (the `zs_td` suite), MASSIVE and the zs_wide workflows
(evaluation only).

## 4. Evaluation

- **Suites** (`benchmark/README.md`; never trained on):
  - `td`: Typed Decisions test split, the 3 training workflows.
  - `zs_td`: zs_td is ONE held-out workflow (security_incidents; dev 300 and final 100 records of the same workflow).
    v1.0.0 was kept on its dev accuracy (A-025) after ~55 logged comparisons on it, so final zs_td estimates accuracy on
    further records of a workflow the model was selected on: it is neither unbiased nor unseen-workflow evidence. Its
    fit-to-fit SD (same code, other RNG) is ~.022 accuracy per fit (SD of a difference .031, 4 pairs), twice its
    record-cluster SE (.011); one workflow gives no between-workflow variance. Unseen-workflow evidence = zs_wide final
    macro_real.
  - `zs_massive`: MASSIVE (en) intents with leak-free option sets. MASSIVE was never trained on, but its label space is
    not new: about 22% of its intent names (13 of 59 development option texts: 1 identical, 12 near) overlap CLINC150 /
    Banking77 option texts in the pool, and 4 final records are exact duplicates of pool records (removed at the next
    benchmark run).
  - `zs_wide`: workflows never trained on: dev 154 workflows (111 from public sources: SNI tasks, SGD services, ABCD; 43
    GLM-synthetic), final 147. The split is by task, not by dataset family: 53 of the 147 final workflows share a family
    with a dev workflow; pool-source tasks were dropped from the final set.
  - `tickets` (in-domain, private) and `tickets_ood` (held-out writing styles of the same generator: a style shift).
- **Metrics**: per-suite mean NLL of the gold option (headline: the mean over td, zs_td, zs_massive, tickets), micro
  accuracy (zs_wide: group-macro over the real sources), ECE over 15 bins with its perfect-calibration floor, certified
  automation (risk .10, δ .10; **this suite only**). Each cell has a 95% record-cluster CI. Scoring: `benchmark/scoring.py`.
- **Selection.** dev numbers are selected (55+ comparisons; every run logged from round 21). The final suites are read
  once per release; zs_td final is not unseen-workflow evidence (above).
- **Leak-free option sets**: for sampled option sets every option is equally likely to be the gold, so option frequency
  reveals nothing (a frequency rule reached .631 micro accuracy against chance .240 on a non-leak-free MASSIVE suite).
- **Rivals** and contamination flags: `benchmark/METHOD.md`.

## 5. Results

**v1.1.0** (final suites, scored once): {{V110_RESULTS_TABLE}}, with per suite NLL, micro accuracy and ECE15 with 95%
record-cluster CIs, the ECE floor (calibrated: yes / no), and certified automation for that suite only. zs_wide final
macro_real {{V110_ZSW_FINAL_MACRO_REAL}}; GLM-synthetic workflows separately {{V110_ZSW_FINAL_GLMSYN}}.

**Calibration status, v1.0.0 development suites** (observed ECE15 vs the 95th percentile of a perfectly calibrated model's
ECE15 on the same suite): td .036 vs .048 and tickets .076 vs .088: **calibrated**; zs_massive .043 vs .041, zs_td .109
vs .032 and zs_wide .080 vs .027: **not calibrated** on those suites.

**v1.0.0 (withdrawn: training pool contained CC BY-NC data).** Final run 2 (2026-10-08; td 300 records / 1,500 questions,
zs_td 100 / 500, zs_massive 1,000 / 1,000, tickets 169 / 507, tickets_ood 147 / 441); no CI was computed for these cells.

| Suite | NLL | Micro accuracy | ECE15 | Certified automation, this suite only (coverage / error) |
|---|---|---|---|---|
| td (seen workflows) | .6502 | .727 | .047 | .342 / .091 |
| zs_td (`security_incidents`, selected on) | 1.1149 | .490 | .105 | 0 |
| zs_massive (leak-free) | .4758 | .826 | .034 | .806 / .079 |
| tickets | .6615 | .702 | .063 | .186 / .021 |
| tickets_ood | .7509 | .642 | .064 | .093 / .053 |

Against the rivals of run 2 (`benchmark/results/run2/RUN2.md`), v1.0.0 trailed the 144M-0.8B rivals and the hosted Jev on
td / zs_td / zs_massive micro accuracy. On each of the three public suites Jev's ECE15 was lower than v1.0.0's, so no
calibration ranking is claimed; no latency or size ranking is claimed either (README).

## 6. Limitations

- **Unseen workflows: low accuracy.** Expect weak decisions on a new workflow until you measure it on labelled cases of
  your own. A few labelled records help mainly by teaching the label prior (README, adaptation table).
- **Fit-to-fit noise.** Two fits of the same code that differ only in their random seed differ by about .03 zs_td micro
  accuracy (SD of a difference); single-fit differences of that size are not evidence (§4).
- **Option-text tilt** on new workflows is reduced by the bias expert (§2.3) but not removed.
- **More data did not simply help**: adding broad text and dialogue data made td, tickets and zs_massive worse (zs_td
  unchanged within noise), and synthetic workflow data did not improve unseen-workflow accuracy; neither is in the
  training data.
- **English only** (base encoder and all data). **Synthetic in-domain data**: the tickets come from one model family
  (Claude Haiku) and have no human labels.
- **Certified automation is per suite**: a threshold certified on one workflow can fail on another (a td-certified
  threshold had error .771 on zs_wide). Certify on labelled rows of the workflow you automate.
- **Python runtime only** (torch + transformers, CPU). Latency is the Python reference on a 4-core CPU, not a phone.
- **Float noise**: batch composition, chunk size and thread count move probabilities by at most about 6e-7 (measured).
- **Base model fetched at first use** from the Hugging Face Hub (or a pre-filled cache), pinned to revision `ffb93f3b`.
- **Reproducibility**: a release is tied to its runtime (sha256 in `config.json`); ej refuses weights packaged for another.

## 7. Intended use

- On-device triage and routing decisions (support tickets, intents, workflow checks) where a probability distribution is
  needed per question and uncertain cases are **escalated** to a slower System 2 (an on-device LLM or a cloud model).
- Ranking, gating and abstention using the returned probabilities, with thresholds certified on your own labelled data.

## 8. Out-of-scope use

- Sole decision-maker for consequential decisions about people (credit, employment, health, legal, safety) without human
  review.
- Non-English text; generative tasks; free-text answers.
- New structured workflows without first measuring quality on labelled examples of that workflow (§6).
- Security-incident triage as a zero-shot claim (that workflow drove model selection).
- Transductive use across workflows, or on option sets that change between records: `adapt` / `observe=True` are for one
  workflow with a fixed option set (§1).

## 9. Licence and attribution

The weights are licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. Adaptations
must be shared under CC BY-SA 4.0 or a compatible licence. Suggested attribution:

> ej 1.1.0 by the ej contributors, licensed under CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/). Derived from intfloat/e5-small-v2 (MIT; Wang et al., arXiv:2212.03533)
> and trained on Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions (Apache-2.0) and an
> in-house support-ticket corpus written with Claude Haiku (not released). Full notices, creators and open questions:
> NOTICE.
