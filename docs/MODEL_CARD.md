# ej 1.0.0: model card

| Field | Value |
|---|---|
| Model | ej 1.0.0 (released 2026-10-08) |
| State key | `14a3e64fb5675f19ab7413e29d24f8fa7fbfa9b656c21a60da03a40d076f28f9` (recorded in `config.json`) |
| Size | **11.2 MB** counted (`size_mb`: bit-level accounting of encoder codes + vocabulary + int8 heads); download 36,078,708 bytes = **36.1 MB** of weight files (`state.safetensors`, `state.json.gz`, `encoder/w23.safetensors`, `encoder/w23.json`: codes are stored unpacked) |
| Latency (CPU, Python reference) | **112.7 ms/record** cold (development-suite probe, 1 CPU thread, prediction-time caches bypassed with `EDGE_COLD=1`); **107 ms/record** mean over the final benchmark suites (1 CPU thread, caches enabled) |
| Language | English only (§6) |
| Licence | weights **CC BY-SA 4.0**; code Apache-2.0 |
| Base model | `intfloat/e5-small-v2` (MIT), revision `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| Weights / code | https://huggingface.co/5ak3t/ej (revision `v1.0.0`) / https://github.com/codecraf8/ej-benchmark-releases |

## 1. What the model does

A **System-1 decision model for devices**: given a `state` (text, or a JSON object as text) and typed questions, it returns
one calibrated probability distribution per question in **one pass of a small encoder plus small heads**, without
decoding tokens. Usage: the repository README.

| Question type | Input | Output |
|---|---|---|
| `choice` | instructions + a runtime list of option texts | a probability per option |
| `noul` | a yes/no statement | P(false), P(true) |
| `score` | instructions + an ordered list of level texts | a distribution over the levels |

- **No LLM at inference**; teachers are used at fit time only and their weights do not ship.
- **Calibrated**: a fitted log-linear pool with per-cell calibration and a lapse term.
- **Record-independent**: no pooling across records at inference.
- Option texts are read as text, so new label spaces can be asked zero-shot; quality on unseen workflows is limited (§6).

## 2. Architecture

**2.1 Low-bit shared encoder (the only transformer on the device).** Base `intfloat/e5-small-v2`, mean pooling,
`query:` / `passage:` prefixes. Trimmed vocabulary of 13,598 WordPiece pieces chosen a priori (a dropped piece is re-split,
never `[UNK]`). **2-bit** embeddings and feed-forward matrices, **3-bit attention**, groups of 128 input columns with fp16
step and offset; GPTQ initialisation, then quantisation-aware distillation to a 4-bit e5 (fixed 30-epoch cosine schedule);
pooled cosine to the 4-bit encoder 0.99176 on held-out training texts. Encoder + vocabulary 8.110 + 0.089 MB. The state is
encoded **once** per record; JSON fields are read by key-aware attention.

**2.2 Expert pool, int8 heads.** 11 expert slots: a deep convex conditional logit over label-agnostic features
(zero-shot capable), wide sparse state-token × option-slot crosses, a rich bilinear/ordinal/MLP scorer, a centred prior-free
expert, field attention, relational evidence over JSON cross-field relations, a slot-free relational reader, a **distilled
decision-encoder scorer**, a **distilled NLI pair head**, and two product-of-experts slots (§2.3). Every head is fitted on
the low-bit encoder's features and compacted to **int8** (per-row int8 matrices; int8 cross weights with 40-bit hashed
keys). An ordinal score expert is built but off (level-text matching did not transfer).

**2.3 Option-text debiasing.** On a new workflow the pooled distribution carried a near-constant tilt from the option
texts. A **bias-only expert** reads the question and the option texts but not the state; the main deep expert is trained as
a product of experts with the frozen bias logits as an offset (Clark et al. 2019; He et al. 2019), and the **negated** bias
expert enters the pool with its own signed weight, so the option-text prior is divided out of the mix.

**2.4 Teachers (fit time only; nothing ships).** NLI cross-encoder `cross-encoder/nli-deberta-v3-xsmall` distilled into the
pair head on a teacher-labelled augmented transfer set of training texts; a decision encoder (e5 layers 10-12 fine-tuned on
the training pool) distilled from its out-of-fold distributions. No large teacher is used.

**2.5 Group-honest stacking.**
- Log-linear pool with lapse per calibration cell (question type × seen/unseen option slots × structured state):
  `p = (1 - eps) softmax(sum_e a_e z_e) + eps / K`; a selective correctness head is adopted per regime only where it beats
  the pool.
- **Group-honest nested cross-fitting**: rows for a held-out group come only from models and teachers that never saw it.
- **Bayesian hierarchical stacking** of the unseen-slot pool (Yao, Pirs, Vehtari & Gelman, arXiv:2101.08954): partial
  pooling root → type → cell with group random effects; the new-group predictive is chosen per regime inside the fit.
- The training pool has 8 groups, so the exact leave-one-group-out path and per-regime selection of the new-group
  predictive are used (grouped K-fold blocks and stacking of the predictives switch on only above 12 groups).

## 3. Training data

One fit on the training pool **v2** (12,719 records, 8 source groups; sha256
`35b11986cdcdda04d3a8dfb430c321a74096b21218eb647cf33470a0897ef67d`). Details and counts: `docs/DATA_CARD.md`.

| Source (groups) | Records | Licence |
|---|---|---|
| Typed Decisions, workflows `agent_trace_observability`, `customer_service`, `invoice_processing` (3) | 736 | Apache-2.0 |
| **Support tickets written by Claude Haiku** for this project, train + calibration splits (1) | 780 | in-house; **not released** |
| Banking77 / CLINC150 `plus` / GoEmotions / Amazon counterfactual (4) | 3,000 / 3,000 / 2,203 / 3,000 | CC BY 4.0 / CC BY 3.0 / Apache-2.0 / CC BY 4.0 |
| **Total** | 12,719 records, 8 groups | |

**Disclosure: Claude Haiku.** The in-domain support tickets were written by Claude Haiku (labels by construction from the
generation spec, no human review). They are part of training; the corpus itself is **not released**, so the training pool
cannot be rebuilt from public data alone.
Not in the training data: Typed Decisions `security_incidents` (held out: the `zs_td` suite) and MASSIVE (evaluation only).

## 4. Evaluation

- **Suites** (never trained on; `benchmark/README.md`): `td` (Typed Decisions test, seen workflows), `zs_td` (workflow
  `security_incidents`, held out = a new task), `zs_massive` (MASSIVE intents, a never-seen label space, leak-free option
  sets), `tickets` (in-domain) and `tickets_ood` (out-of-distribution tickets).
- **Metrics**: per-suite mean NLL of the gold option (headline: the mean over td, zs_td, zs_massive, tickets), accuracy,
  ECE over 15 bins, certified automation (risk .10, δ .10). Scoring code: `benchmark/scoring.py`.
- **Leak-free option sets**: for sampled option sets every option is equally likely to be the gold, so option frequency
  reveals nothing (a frequency rule reached .631 accuracy against chance .240 on a non-leak-free MASSIVE suite).
- **Rivals** and contamination flags: `benchmark/METHOD.md`.

## 5. Results

Final benchmark, run 2 (2026-10-08; sealed test suites, scored once; td 300 records / 1,500 questions, zs_td 100 / 500,
zs_massive 1,000 / 1,000, tickets 169 / 507, tickets_ood 147 / 441). Full table with rivals and flags:
`benchmark/results/run2/RUN2.md`.

| Suite | NLL | Accuracy | ECE15 | Certified automation (coverage / error) |
|---|---|---|---|---|
| td (seen workflows) | .6502 | .727 | .047 | .342 / .091 |
| zs_td (`security_incidents`, held out) | 1.1149 | .490 | .105 | 0 |
| zs_massive (leak-free) | .4758 | .826 | .034 | .806 / .079 |
| tickets | .6615 | .702 | .063 | .186 / .021 |
| tickets_ood | .7509 | .642 | .064 | .093 / .053 |
| **Mean (td, zs_td, zs_massive, tickets)** | **.7256** | | **.062** | |

Published reference points on the same 400 Typed Decisions test cases (td + zs_td): laya 0.766, Jev 0.727. In run 2,
ej is the smallest and fastest model in the table and has the lowest mean ECE; it trails the 144M-0.8B rivals and the
hosted Jev on td / zs_td / zs_massive accuracy (`benchmark/results/run2/RUN2.md`).

## 6. Limitations

- **Unseen workflows: low accuracy.** On the held-out workflow (zs_td) this model scores accuracy **.490**; rivals that
  probably saw that workflow score .71-.77. zs_td is one workflow of 100 records (500 questions). Expect weak decisions on a
  new workflow until you measure it on labelled cases of your own.
- **Fit-to-fit noise on that number is about .05.** Fitting identical code on identical data in two pipelines gave
  development zs_td accuracy .466 vs .419; zs_td is a single workflow, so differences of this size are not evidence.
- **Option-text tilt** on new workflows is reduced by the debiasing expert (§2.3) but not removed.
- **More data did not simply help**: adding broad text and dialogue data made every development suite worse, and
  synthetic workflow data did not improve unseen-workflow accuracy; neither is in this model's training data.
- **English only** (base encoder and all data). **Synthetic in-domain data**: the tickets come from one model family
  (Claude Haiku) and have no human labels.
- **Selection optimism**: the model was selected on development suites; the sealed test suites are the unbiased read.
- **Certified automation is suite-dependent**: .806 of zs_massive and .342 of td questions are covered at a certified
  10% error bound, but 0 on zs_td and .09-.19 on the ticket suites (§5).
- **Python runtime only** (torch + transformers, CPU). Latency is the Python reference on a 4-core CPU, not a phone.
- **Batch composition** moves probabilities at the float-noise level (≈1e-7) because records are padded together.
- **Base model fetched at first use** from the Hugging Face Hub (or a pre-filled cache), pinned to revision `ffb93f3b`.
- **Reproducibility**: a release is tied to its runtime (sha256 in `config.json`); ej refuses weights packaged for another.

## 7. Intended use

- On-device triage and routing decisions (support tickets, intents, workflow checks) where a calibrated distribution is
  needed per question and uncertain cases are **escalated** to a slower System 2 (an on-device LLM or a cloud model).
- Ranking, gating and abstention using the returned probabilities, with thresholds certified on your own labelled data.

## 8. Out-of-scope use

- Sole decision-maker for consequential decisions about people (credit, employment, health, legal, safety) without human
  review.
- Non-English text; generative tasks; free-text answers.
- New structured workflows without first measuring quality on labelled examples of that workflow (§6).
- Security-incident triage as a zero-shot claim (that workflow is the held-out test workflow).
- Transductive use that pools option statistics across records to improve accuracy.

## 9. Licence and attribution

The weights are licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. Adaptations
must be shared under CC BY-SA 4.0 or a compatible licence. Suggested attribution:

> ej 1.0.0 by the ej contributors, licensed under CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/). Derived from intfloat/e5-small-v2 (MIT; Wang et al., arXiv:2212.03533)
> and trained on Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions (Apache-2.0), Amazon
> counterfactual (CC BY 4.0) and an in-house support-ticket corpus written with Claude Haiku (not released); distillation
> signal from cross-encoder/nli-deberta-v3-xsmall (Apache-2.0; trained on SNLI and MultiNLI). Full notices: NOTICE.
