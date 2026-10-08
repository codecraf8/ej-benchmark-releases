# ej

**ej** is a small on-device model for **typed decisions**. You give it a `state` (free text, or a JSON object as text) and a
set of typed questions; it returns **one calibrated probability distribution per question, in one pass**, without
generating tokens and without an LLM at inference.

| Question type | You give | You get |
|---|---|---|
| `choice` | instructions + a list of option texts (any labels, chosen at run time) | P(option) for each option |
| `noul` | a yes/no statement, options `false` / `true` | [P(false), P(true)] |
| `score` | instructions + ordered level texts | a distribution over the levels |

- **Small**: 11.2 MB counted model size (2-bit encoder with 3-bit attention, int8 heads); 36.1 MB of weight files to
  download, plus the public base model `intfloat/e5-small-v2` (134 MB, fetched once from the Hugging Face Hub).
- **Fast on a CPU**: 107 ms per record on average over the benchmark suites (Python, 1 CPU thread, caches enabled);
  112.7 ms per record cold (development-suite probe, 1 thread, prediction-time caches bypassed).
- **Calibrated probabilities are the product**: automate confident cases, escalate the rest to a slower system.
- **Pickle-free weights**: safetensors plus a JSON skeleton, decoded with a class allowlist, every file sha256-checked.
- Weights: Hugging Face **[`5ak3t/ej`](https://huggingface.co/5ak3t/ej)**, revision `v1.0.0` (CC BY-SA 4.0). Code: this repository (Apache-2.0).

## Install

```bash
pip install git+https://github.com/codecraf8/ej-benchmark-releases
```

Python >= 3.10 (tested on 3.11.15, CPU). Dependencies are pinned to the tested versions (`pyproject.toml`: torch 2.5.1,
transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface_hub 1.33.0, numpy 2.2.6).

## Quickstart

```python
import ej

model = ej.load('5ak3t/ej', revision='v1.0.0')   # or ej.load('/path/to/weights-dir')
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

Or run `python examples/quickstart.py [--weights DIR_OR_REPO] [--revision REV] [--records file.jsonl]`, which predicts
`ej.EXAMPLE_RECORD` (the record above with four teams and two more state fields) and prints:

```
route (choice)
  0.302  returns  <- argmax
  0.211  billing
  0.263  tech
  0.224  sales
needs_human (noul)
  0.600  false  <- argmax
  0.400  true
urgency (score)
  0.320  0
  0.420  1  <- argmax
  0.260  2
```

This is a made-up workflow the model never saw, and the distributions are correspondingly flat.

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
- `model.predict` validates records first (`ej.validate_records`, raises `ej.RecordError`) and encodes them in one batch.

## Output format

`model.predict(records)` returns one dict per record, `{qid: [p_1, ..., p_K]}`, each list in the order of that
question's `options`, plain Python floats in [0, 1] summing to 1 (within 1e-9).

## Adapting to your workflow

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
(`'answers': {qid: key}`), or by option index in the benchmark format (`'gold': {qid: {'label': i}}`). Measured on our
dev suites (group-macro accuracy, sources weighted equally; paired bootstrap 95% CIs; 8 labelled records per workflow):

| setting | 154 unseen workflows (public sources) | 1 unseen workflow (`zs_td` dev) | 3 seen workflows (`td` dev) |
|---|---|---|---|
| 8 labelled examples vs zero-shot | .341 → .397, +.056 [+.034, +.080] | .465 → .532, +.067 [+.052, +.083] | +.011 [+.004, +.018] |
| 16 labelled examples vs zero-shot | (workflows have ≤ 12 records) | .465 → .548, +.083 [+.065, +.101] | +.010 [+.001, +.020] |
| unlabelled only vs zero-shot (experimental) | .364 → .378, +.014 [+.003, +.027] | +.005 [−.005, +.015] | +.012 [−.004, +.030] |
| 8 labelled + unlabelled vs 8 labelled (experimental) | −.001 [−.020, +.013] | +.011 [+.006, +.016] | −.003 [−.009, +.003] |

Labelled examples also lower the log loss (unseen workflows: −.072 NLL at 8 examples, −.153 on `zs_td` at 16). The
label-free path is **experimental**: its gain on unseen workflows is small (below +.02), and on seen workflows it is not
significant. Fit time on one CPU thread: 15 ms (16 labelled records), 19 ms (100 unlabelled), 29 ms (both); an `observe`
batch of 10 records adds about 5 ms on top of its prediction pass, which `predict(..., observe=True)` reuses.

## Weights, integrity and caches

The weights directory (Hugging Face repo, or `scripts/hf_layout.py` output) holds `config.json`, `state.safetensors`,
`state.json.gz`, `encoder/w23.safetensors`, `encoder/w23.json`, `README.md` (model card), `NOTICE`, `LICENSES/` and
`SHA256SUMS`. `ej.load` checks, before decoding anything: every file's sha256 against `config.json`; the files against the
known-release hashes shipped in `ej.integrity` (warning if the state is not listed); the runtime modules against
`src/ej/_runtime/RUNTIME_SHA256` and the runtime hash recorded in `config.json`; the e5 revision. Decoding never unpickles.
`intfloat/e5-small-v2` is pinned to revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`. Caches: `$HF_HOME` (Hugging Face
default) and `$EDGE_CACHE` (default `~/.cache/ej`, a 93 KB trimmed vocabulary). Offline use: `HF_HUB_OFFLINE=1` with a
filled cache.

## Benchmark

Final benchmark (run 2, 2026-10-08) on sealed test suites. Full table, per-suite NLL / ECE / certified automation and
fairness flags: [benchmark/results/run2/RUN2.md](benchmark/results/run2/RUN2.md).

| model | size | td acc | zs_td acc | zs_massive acc | tickets acc | tickets_ood acc | mean NLL (4) | mean ECE (4) | ms/record |
|---|---|---|---|---|---|---|---|---|---|
| **ej 1.0.0** | 11.2 MB | .727 | .490 | .826 | .702 | .642 | .7256 | .062 | 107 |
| Jev 1.13.0 (hosted API) | closed | .737 | .742 | .957 | .730 | .626 | .786 | .080 | 323 (network) |
| laya | 421M params | .766 | .766 | .866 | .669 | .617 | .6975 | .169 | 12,666 |
| kev-0.8b | 0.8B | .415 | .520 | .895 | .679 | .608 | .806 | .083 | 8,979 |
| OpenThai-SystemOne | 0.75B | .515 | .632 | .971 | .659 | .617 | .846 | .157 | 7,404 |
| Julia-1 (llama.cpp BF16) | 144M | .733 | .706 | .792 | .513 | .465 | 1.613 | .231 | 3,718 |
| gliclass-edge v3.0 | 33M | .367 | .484 | .545 | .304 | .367 | 2.157 | .372 | 651 |

Mean NLL / ECE over td, zs_td, zs_massive and tickets; ms/record on one CPU thread (td suite for rivals).
Flags: laya trained on the `security_incidents` workflow (zs_td is not zero-shot for it), Julia-1 and Jev likely saw it;
OpenThai trained on MASSIVE intents; ej trained on the training split of the ticket corpus (tickets is in-domain for ej
only). zs_td is one workflow of 100 records (500 questions), and two fits of identical code differ by about .05 there,
so small differences on that suite are within fit-to-fit variation.

**How it works.** `benchmark/run_bench.py` gives every model the same blind records and scores the returned
distributions with `benchmark/scoring.py` (NLL, accuracy, ECE, certified automation). Suites: `td` and `zs_td` (Typed
Decisions test split; `zs_td` is a workflow held out of ej's training) and `zs_massive` (MASSIVE intents with leak-free
option sets) are built from **public** datasets; `tickets` and `tickets_ood` come from a **private** ticket corpus. No
suite file is distributed here, so the code is the method reference; it runs on any labelled suite in the ej format.
Details: [benchmark/README.md](benchmark/README.md), [benchmark/METHOD.md](benchmark/METHOD.md).

## Limitations

- **Unseen workflows: low accuracy** (zs_td .490). Measure on your own labelled cases before automating; differences of
  about .05 on that suite are within fit-to-fit variation. A few labelled records help (`model.adapt`, above).
- **English only.** **Python runtime only** (torch + transformers, CPU); no mobile/native runtime in v1.
- **Synthetic in-domain data**: the support tickets were written by Claude Haiku (no human labels).
- The base model is downloaded from the Hugging Face Hub on first use.
- The runtime is imported by bare module names (`student`, `student_lb`, ...) from `ej/_runtime`; a module of your own
  with one of those names conflicts (ej refuses to load). Importing it sets `torch.set_num_threads(2)`.
- Batch composition moves probabilities at the float-noise level (≈1e-7).
- More: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) §6, [docs/DATA_CARD.md](docs/DATA_CARD.md) §7.

## Repository layout

| path | what |
|---|---|
| `src/ej/` | the package: `load`, `Model.predict`, `Model.adapt` (`adapt.py`, `adapt_math.py`), records, loader, integrity checks, pickle-free codec |
| `src/ej/_runtime/` | the prediction code: research modules (code unchanged, comments cleaned), sha256-pinned ([README](src/ej/_runtime/README.md)) |
| `examples/quickstart.py`, `tests/` | example; tests (`pytest`; set `EJ_WEIGHTS_DIR` to run the prediction tests) |
| `scripts/hf_layout.py` | builds the Hugging Face upload directory from weight files (never uploads) |
| `docs/` | model card, data card |
| `benchmark/` | benchmark code, method notes, aggregate results |

## Licence and attribution

- Code: **Apache-2.0** (`LICENSE`). Weights: **CC BY-SA 4.0** (`LICENSES/CC-BY-SA-4.0.txt`); adaptations must be shared
  under CC BY-SA 4.0 or a compatible licence. Base model `intfloat/e5-small-v2`: MIT (`LICENSES/MIT-e5-small-v2.txt`).
- Training data: Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions (Apache-2.0),
  Amazon counterfactual (CC BY 4.0), an in-house ticket corpus written with Claude Haiku (not released). Full notices:
  `NOTICE`, `docs/MODEL_CARD.md` §9.
- "Jev" and "System One" are names used by TypeSafe.ai; ej is not affiliated with or endorsed by TypeSafe.ai.

## Citation

```bibtex
@software{ej_2026,
  title   = {ej: calibrated typed decisions on device},
  author  = "{The ej contributors}",
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/codecraf8/ej-benchmark-releases},
  note    = {Weights: https://huggingface.co/5ak3t/ej, revision v1.0.0 (CC BY-SA 4.0)}
}
```
