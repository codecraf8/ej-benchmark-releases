# ej

**ej** is a small on-device model for **typed decisions**. You give it a `state` (free text, or a JSON object as text) and a
set of typed questions; it returns **one probability distribution per question, in one pass**, without generating tokens
and without an LLM at inference.

> **Status (2026-10-08): the weights are not yet public** (the Hugging Face repository `5ak3t/ej` is private: HTTP 401
> without access). **v1.0.0 was withdrawn before publication**: its training pool contained 3,000 Amazon counterfactual
> records, licensed **CC BY-NC 4.0** upstream, incompatible with CC BY-SA 4.0 weights (`NOTICE`).
> **v1.0.1 is the licence fix of the withdrawn v1.0.0; same architecture**: the same code refitted on the licence-clean
> pool v2b (v2 without that source), the low-bit encoder re-distilled on v2b texts. It is a private Hugging Face revision
> (tag `v1.0.1`); publication is **pending** the maintainer's decision. A later round of model changes failed its
> pre-registered evaluation and is not released. This repository is the code (Apache-2.0); `ej.load` takes a local
> weights directory, and none is distributed here.

| Question type | You give | You get |
|---|---|---|
| `choice` | instructions + a list of option texts (any labels, chosen at run time) | P(option) for each option |
| `noul` | a yes/no statement, options `false` / `true` | [P(false), P(true)] |
| `score` | instructions + ordered level texts | a distribution over the levels |

- **Size, four measures** (decimal MB unless marked; details in `docs/MODEL_CARD.md`): *counted* = a bit-level bound
  (2-bit encoder with 3-bit attention, int8 heads), not a file format; *on disk* = the weights directory; *download* =
  what a first `ej.load` fetches; *resident* = tensors held while predicting. v1.0.1: counted 10.87 MiB (11.40 MB), on
  disk 35.2 MB, download about 169.6 MB (incl. the full base model `intfloat/e5-small-v2`, 134 MB), resident 110.4 MB
  (state + dequantised low-bit encoder; the runtime also keeps the fp32 base model it builds, about 133 MB, audit A1-1).
- **Latency** (one torch thread, one record per call, measured with the benchmark runner): [Latency](#latency).
- **Calibration is measured per suite**, not assumed: calibrated only where the ECE lies within the range a perfectly
  calibrated model would show there ([glossary](#glossary)). Automate confident cases and escalate the rest.
- **Pickle-free weights**: safetensors plus a JSON skeleton, decoded with a class allowlist, every file sha256-checked;
  after loading, no runtime module can unpickle anything.

## Install

```bash
pip install git+https://github.com/codecraf8/ej-benchmark-releases
```

Python >= 3.10 (tested on 3.11.15, CPU). Dependencies are pinned to the tested versions (`pyproject.toml`: torch 2.5.1,
transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface_hub 1.33.0, numpy 2.2.6). The tests that need no
weights run with `pip install -e '.[test]'` and `pytest`.

## Quickstart

You need an ej weights directory (the `scripts/hf_layout.py` layout). No public one exists yet (status above).

```python
import ej

model = ej.load('/path/to/ej-weights')   # a local weights directory
record = {
    'state': '{"customer_tier": "gold", "channel": "email", "message": "The blender arrived with a cracked jug. '
             'I want a replacement before Friday, otherwise refund me."}',
    'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team should handle this request?',
                  'options': [{'key': 'returns', 'text': 'Returns and replacements for damaged or wrong items'},
                              {'key': 'billing', 'text': 'Billing, invoices and payment problems'}]},
        'needs_human': {'type': 'noul', 'instructions': 'The customer is upset enough that a human agent should reply.',
                        'options': ej.NOUL_OPTIONS},
        'urgency': {'type': 'score', 'instructions': 'How urgent is this request?',
                    'options': [{'key': '0', 'text': 'Low: can wait a week'},
                                {'key': '1', 'text': 'Medium: answer within two days'},
                                {'key': '2', 'text': 'High: answer today'}]},
    },
}
(probs,) = model.predict([record])
print(probs['route'])          # [p_returns, p_billing], sums to 1
```

`python examples/quickstart.py --weights /path/to/ej-weights [--records file.jsonl]` predicts `ej.EXAMPLE_RECORD` (the
record above with four teams and two more state fields) and prints each distribution. With the withdrawn v1.0.0 weights
the route distribution was flat (.302 / .211 / .263 / .224): a made-up workflow the model never saw.

## Input format

```json
{"id": "optional", "state": "text, or a JSON object serialised as text",
 "questions": {"<qid>": {"type": "choice | noul | score", "instructions": "text",
                         "options": [{"key": "k1", "text": "short description"}, {"key": "k2", "text": "..."}]}}}
```

- At least 2 options per question; keys unique within a question. `noul` options are exactly `false` and `true`
  (`ej.NOUL_OPTIONS` gives the standard texts). `score` options are ordered levels, lowest first.
- A JSON object as `state` is read field by field (key-aware attention over fields); plain text works too.
- Option texts are read as text: write them as short descriptions, as above. Other fields (`id`, `source`) are ignored.
- `model.predict(records, threads=None, chunk_size=None)` validates records first (`ej.validate_records`, raises
  `ej.RecordError`), then encodes them in chunks of `chunk_size` records (default 64; `0` = one batch).

## Output format

`model.predict(records)` returns one dict per record, `{qid: [p_1, ..., p_K]}`, each list in the order of that
question's `options`, plain Python floats in [0, 1] summing to 1 (within 1e-9).

**Determinism and float noise.** Predictions are deterministic for a fixed thread count and chunking. The records padded
together, the chunk size and the thread count move probabilities only at float-noise level: measured max |Δp| 5.5e-7
(chunks of 8 vs one batch) and 6.5e-7 (1 vs 2 threads) on 72 development records; 2 threads and one batch reproduce the
reference predictions exactly.

**Process hygiene.** `ej.load` leaves torch's thread count, its random state and `transformers` unchanged (the runtime's
import-time `set_num_threads(2)` and `manual_seed(0)` are undone). `predict` uses `threads` torch threads (default: the
model's `threads`, 2, set by `ej.load(..., threads=2)`) only inside the call and restores the previous count afterwards;
`model.last_threads` reports the count in effect during the last call. The encoder memo is bounded
(`ej.load(..., memo_max=20000)` texts, least recently used evicted; about 21 KB per record); `EDGE_COLD=1` disables it.

## Adapting to your workflow

**Record independence.** Zero-shot `predict` is record-independent: each record's output depends only on that record (up
to the float noise above). `adapt` and `predict(..., observe=True)` are an opt-in, per-workflow transductive mode: they
pool option statistics across that workflow's records, so use them only for one workflow with a fixed option set (the same
question ids and option keys).

A zero-shot model cannot know how often each answer occurs in your workflow. `model.adapt` learns one logit offset per
(question id, option key) from that workflow's own records and returns a model that predicts the same way (no fine-tuning,
a fit of a few milliseconds plus one prediction pass over the records you give it). Use one adapted model per workflow.

```python
labelled = [{**r, 'answers': {'route': 'returns', 'needs_human': 'true', 'urgency': '2'}} for r in my_records[:8]]
adapted = model.adapt(examples=labelled)      # a few labelled records: the option tilt
probs = adapted.predict(new_records)

adapted = model.adapt(unlabeled=past_requests)                   # EXPERIMENTAL: no labels at all
probs = adapted.predict(todays_requests, observe=True)           # keeps learning from traffic, stores no request
```

`model.adapt()` with neither argument returns the model itself (identical predictions). Answers are given by option key
(`'answers': {qid: key}`), or by option index in the benchmark format (`'gold': {qid: {'label': i}}`).

**Measured with the withdrawn v1.0.0 weights** on development suites (v1.0.1: second table below). k = labelled
**records** per workflow (every question of a support record is labelled; the labelled answers are in brackets).
Group-macro accuracy (mean over workflows within a source, sources weighted equally); 95% CIs: a t interval over
workflows when a suite has 10 or more workflows, else a record-cluster bootstrap (records resampled, the 3 support-set
salts averaged inside each draw). The count prior is the no-model baseline: add-one counts of the labelled answers.

| setting | 154 workflows, zs_wide dev (111 public-source + 43 GLM-synthetic) | 1 workflow, zs_td dev (selected on, see glossary) | 3 seen workflows, td dev (selection data) |
|---|---|---|---|
| k = 8 vs zero-shot | .341 → .397, +.056 [+.031, +.081] (153 workflows; about 16.6 answers) | .465 → .532, +.067 [+.047, +.089] (40 answers) | +.011 [+.001, +.022] (40 answers) |
| k = 16 vs zero-shot | not measured: these workflows have at most 12 records | .465 → .548, +.083 [+.058, +.110] (80 answers) | **+.010 [−.002, +.023]: covers 0** (80 answers) |
| k-shot vs the count prior | **+.001 [−.007, +.009] at k = 8: no gain beyond the label prior** | +.008 [+.000, +.017] at k = 16 | +.022 [+.003, +.043] at k = 16 |

**v1.0.1 weights**, same protocol, development suites (`benchmark/results/adapt-v1.0.1/adapt_eval.json`):

| setting (v1.0.1) | 154 workflows, zs_wide dev | 1 workflow, zs_td dev (selected on) | 3 seen workflows, td dev |
|---|---|---|---|
| k = 8 vs zero-shot | .356 → .389, **+.032 [−.011, +.076]: covers 0** (153 workflows; about 16.6 answers) | .396 → .530, +.134 [+.108, +.160] (40 answers) | +.008 [−.003, +.020]: covers 0 (40 answers) |
| k = 16 vs zero-shot | not measured: these workflows have at most 12 records | .396 → .545, +.149 [+.118, +.183] (80 answers) | +.016 [+.003, +.029] (80 answers) |
| k-shot vs the count prior | +.008 [−.0001, +.016] at k = 8: covers 0 | +.004 [+.001, +.007] at k = 16 | +.028 [+.012, +.046] at k = 16 |
| label-free (experimental) vs zero-shot | +.009 [−.014, +.031]: covers 0 | +.011 [−.002, +.025]: covers 0 | +.010 [−.007, +.027]: covers 0 |

How to read it: on unseen workflows what the labelled records teach is mostly the workflow's **label prior** (how often
each answer occurs); the add-one count prior alone reaches the same group-macro accuracy (.3958 vs .3968 at k = 8). The
zs_wide figures weight the 43 GLM-synthetic workflows as one source of four (1/4 of the weight), and their gold labels have
not been independently checked; the adaptation evaluator reports the four-source macro only, so a macro over the three
real sources alone (sni, sgd, abcd) is not available for these tables. Labelled examples also lower the log loss (v1.0.0:
zs_wide −.072 NLL at k = 8, zs_td −.153 at k = 16; v1.0.1: −.091 [−.112, −.071] and −.175 [−.194, −.156]).

The **label-free path is experimental**: with v1.0.0 its measured gain on unseen workflows was below +.02, and its
intervals came from the retired question-level bootstrap, so they are not repeated here; it is re-measured with
the intervals above for v1.0.1 (last row of its table: no gain distinguishable from 0 on any suite). Fit time on one CPU thread: 15 ms (16 labelled
records), 19 ms (100 unlabelled), 29 ms (both); an `observe` batch of 10 records adds about 5 ms on top of its prediction
pass, which `predict(..., observe=True)` reuses.

## Weights, integrity and caches

The weights directory (`scripts/hf_layout.py` output; the Hugging Face repository has the same files) holds `config.json`,
`state.safetensors`, `state.json.gz`, `encoder/w23.safetensors`, `encoder/w23.json`, `README.md` (model card), `NOTICE`,
`LICENSES/` and `SHA256SUMS`. `ej.load` checks, before decoding anything: every file's sha256 against `config.json`; the
files against the known-release hashes shipped in `ej.integrity` (warning if the state is not listed); the runtime
modules against `src/ej/_runtime/RUNTIME_SHA256` and the runtime hash recorded in `config.json`; the e5 revision.
Decoding never unpickles, and after loading every runtime module's `torch.load` / `pickle.load` refuses
(`ej.scope.forbid_unpickling`). The maintainer-only export of the original research pickles is
`scripts/maintainer_pickle.py`, outside the package, behind `EJ_MAINTAINER_UNPICKLE=1` and sha256 checks.
`intfloat/e5-small-v2` is pinned to revision `ffb93f3bd4047442299a41ebb6fa998a38507c52` inside ej calls only. Caches:
`$HF_HOME` (Hugging Face default) and `$EDGE_CACHE` (default `~/.cache/ej`, a 93 KB trimmed vocabulary). Offline use:
`HF_HUB_OFFLINE=1` with a filled cache.

**Provenance.** `config.json` names `code_commit`, a commit of this repository that `scripts/hf_layout.py` records only
when the tree is clean and the commit is contained in a pushed branch, and `research_commit`, the research tree whose code
and training pool reproduce `state_key` (`python scripts/state_key.py --git REPO COMMIT POOL --expect KEY`). For the
withdrawn v1.0.0 state `14a3e64f...`: research commit `f46cf7c` (private repository codecraf8/rev, pushed branch
`edge-master`) with pool v2 reproduces the key; its packaging `config.json` named an unreachable commit (`788789e`), which
`ej.integrity.RELEASE_PROVENANCE` corrects (runtime commit `3157bb9` of this repository). v1.0.1 (`3b3e66d2...`): `f46cf7c`
with `--ck-dir lowbit-b3b010513f948ceb` and pool v2b reproduces the key; its `research_commit` is a later research head
tested to predict the state exactly as `f46cf7c` does. A state loads by the key recorded in its own files.

## Benchmark

Every model gets the same blind records (`benchmark/run_bench.py`) and is scored with `benchmark/scoring.py` (NLL,
accuracy, ECE15, certified automation). Suites, builders and sha256: [benchmark/README.md](benchmark/README.md) and
[benchmark/suites/](benchmark/suites/); method and contamination flags: [benchmark/METHOD.md](benchmark/METHOD.md).

**v1.0.1.** Final test suites, scored once for the release (2026-10-08; run 3,
[benchmark/results/run3/RUN3.md](benchmark/results/run3/RUN3.md)). Each cell carries a 95% record-cluster CI; each
ej-vs-rival difference has a paired CI in RUN3.md, and an ordering is stated only where that CI excludes 0.

| final suite | records / questions | NLL [95% CI] | micro accuracy [95% CI] | ECE15 [95% CI]; calibrated on this suite? |
|---|---|---|---|---|
| td (seen workflows) | 300 / 1,500 | .663 [.613, .716] | .721 [.695, .746] | .071 [.050, .094]; no |
| zs_td (`security_incidents`, selected on) | 100 / 500 | 1.145 [1.113, 1.180] | .422 [.388, .452] | .094 [.068, .135]; no |
| zs_massive (leak-free) | 1,000 / 1,000 | .481 [.430, .530] | .832 [.808, .857] | .051 [.041, .073]; no |
| tickets (in-house) | 169 / 507 | .629 [.583, .680] | .712 [.679, .746] | .033 [.028, .080]; yes |
| tickets_ood (in-house, style shift) | 147 / 441 | .715 [.650, .787] | .683 [.639, .726] | .050 [.039, .099]; yes |

The unseen-workflow read is zs_wide final (147 workflows never trained on, read once), group-macro accuracy over the real
sources **macro_real .419 [.379, .458]** (105 workflows: SNI .456, SGD .571, ABCD .229; t interval over workflows within
sources). GLM-synthetic workflows reported separately: .358 [.323, .394] (42 workflows). Rival comparisons with paired
CIs and the seed variability of this code (six fits: SD .0105 in the development metric) are in RUN3.md.

**v1.0.0 (withdrawn: training pool contained CC BY-NC data).** Its run-2 table stays in [RUN2.md](benchmark/results/run2/RUN2.md)
with the round-20 audit corrections (no CIs; rival latencies not comparable; rankings withdrawn).

**Development numbers are selected.** dev numbers are selected (55+ comparisons; every run logged from round 21): the
development suites were used to choose among many candidates, so their numbers are optimistic. zs_td is one workflow that
drove selection (glossary); the unseen-workflow evidence is zs_wide final, read once.

## Latency

Measured with `benchmark/run_bench.py` (one record per call after a warm-up record, wall clock, 1 torch thread set inside
each call and checked: `threads_measured` = [1]) on the development suites. v1.0.1 runs the same runtime code as v1.0.0
and was re-measured on td (first row; 1-minute load average 1.02-1.10); the other rows used the withdrawn v1.0.0 weights.
Box: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, Python 3.11.15, torch 2.5.1 CPU; 1-minute load average
0.83-1.24 during the runs; 2026-10-08.

| development suite | records | warm: mean / median ms per record | cold (`EDGE_COLD=1`): mean / median ms per record |
|---|---|---|---|
| td, v1.0.1 weights | 164 | 356.6 / 363.1 | 1,922.0 / 1,413.7 |
| td (JSON states, 5 questions) | 164 | 421.9 / 394.2 | 2,123.9 / 1,562.6 |
| zs_td (JSON states, 5 questions) | 300 | 460.5 / 449.9 | 1,398.5 / 1,359.1 |
| zs_massive (short text, 1 question) | 1,000 | 95.7 / 88.7 | 282.9 / 279.6 |
| tickets (text, 3 questions) | 104 | 301.2 / 307.8 | 693.7 / 685.7 |

Warm = prediction-time caches on (texts seen in earlier records, such as option texts, are not re-encoded); cold = every
text encoded again. Summaries: `benchmark/results/latency-r21/`. Command: `EJ_WEIGHTS=DIR BENCH_THREADS=1 [EDGE_COLD=1]
python benchmark/run_bench.py dev/<suite>.jsonl ej`. Batching is faster per record (one call over many records shares
the encoder passes); the 107 ms published for v1.0.0 was such a batched mean on 2 threads, withdrawn.

Rival latencies in RUN2.md were measured on the final suites on another day under unrecorded load, so they are **not
comparable** with these numbers and are not ranked against them.

## Glossary

- **k (few-shot)**: the number of labelled *records* per workflow; the labelled answers are k × questions per record.
- **micro accuracy**: share of questions answered correctly, pooled over a suite. **group-macro accuracy**: mean over
  workflows within a source, then over sources (equal weight). **macro_real**: group-macro over the real sources only
  (zs_wide: sni, sgd, abcd; the GLM-synthetic source is reported separately).
- **zs_td**: one held-out Typed Decisions workflow (`security_incidents`). It was used for model selection (model card
  §4), so its numbers are not unseen-workflow evidence.
- **zs_wide**: development and final suites of many workflows (public-source tasks plus GLM-synthetic ones), never trained
  on.
- **tickets_ood**: tickets in held-out *writing styles* from the same generator and departments: a writing-style shift,
  not a new domain.
- **calibrated on a suite**: the observed ECE15 is at or below the 95th percentile of the ECE15 that a perfectly
  calibrated model with the same confidences would show on that suite; otherwise "not calibrated on that suite".
- **certified automation (CA)**: coverage at a confidence threshold certified on half of a suite to keep the error at or
  below 10%. It holds for **this suite only**: a threshold certified on one workflow does not transfer to another.

## Limitations

- **Unseen workflows: low accuracy.** Measure on your own labelled cases before automating. A few labelled records help
  (`model.adapt`), mostly by teaching the workflow's label prior.
- **English only.** **Python runtime only** (torch + transformers, CPU); no mobile/native runtime.
- **Synthetic in-domain data**: the support tickets were written by Claude Haiku (no human labels).
- The base model is downloaded from the Hugging Face Hub on first use.
- The runtime is imported by bare module names (`student`, `student_lb`, ...) from `ej/_runtime`; a module of your own
  with one of those names conflicts (ej refuses to load). The runtime still ships its fit-time modules (unused at
  prediction and unable to unpickle); they leave the package with a later runtime.
- More: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) §6, [docs/DATA_CARD.md](docs/DATA_CARD.md) §7.

## Repository layout

| path | what |
|---|---|
| `src/ej/` | the package: `load`, `Model.predict`, `Model.adapt` (`adapt.py`, `adapt_math.py`), records, loader, integrity checks, process scope (`scope.py`), pickle-free codec |
| `src/ej/_runtime/` | the prediction code: research modules (code unchanged, comments cleaned), sha256-pinned ([README](src/ej/_runtime/README.md)) |
| `examples/quickstart.py`, `tests/` | example; tests (`pytest`; set `EJ_WEIGHTS_DIR` to run the prediction tests) |
| `scripts/` | `hf_layout.py` (builds the upload directory, never uploads), `state_key.py` (recomputes a state key), `maintainer_pickle.py` (maintainer-only), `check_file_length.py` |
| `docs/` | model card, data card |
| `benchmark/` | benchmark code, suite builders and sha256, method notes, aggregate results |
| `.github/workflows/tests.yml` | CI: the tests that need no weights, on pull requests and on manual dispatch (no deployment) |

## Licence and attribution

- Code: **Apache-2.0** (`LICENSE`). Weights: **CC BY-SA 4.0** (`LICENSES/CC-BY-SA-4.0.txt`); adaptations must be shared
  under CC BY-SA 4.0 or a compatible licence. Base model `intfloat/e5-small-v2`: MIT (`LICENSES/MIT-e5-small-v2.txt`).
- Training data of v1.0.1: Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions
  (Apache-2.0), an in-house ticket corpus written with Claude Haiku (not released); NLI teacher cross-encoder/nli-deberta-v3-xsmall
  (Apache-2.0). Amazon counterfactual (CC BY-NC 4.0) was in the withdrawn v1.0.0 pool only. Details and open questions: `NOTICE`.
- "Jev" and "System One" are names used by TypeSafe.ai; ej is not affiliated with or endorsed by TypeSafe.ai.

## Citation

```bibtex
@software{ej_2026,
  title   = {ej: calibrated typed decisions on device},
  author  = "{The ej contributors}",
  year    = {2026},
  version = {1.0.1},
  url     = {https://github.com/codecraf8/ej-benchmark-releases},
  note    = {Code only; the weights are not yet public}
}
```
