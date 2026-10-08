# Changelog

## 1.0.1 (2026-10-08): licence fix of the withdrawn v1.0.0; same architecture

Weights: **v1.0.1** = the v1.0.0 model code (same architecture, same runtime modules: `RUNTIME_SHA256` unchanged) refitted
on the licence-clean pool v2b, with the low-bit encoder re-distilled on pool v2b texts (state `3b3e66d28fb423f9`, encoder
`lowbit-b3b010513f948ceb`). It is a licence fix, not a model improvement: the final suites were scored once for it
(`benchmark/results/run3/RUN3.md`). It is packaged as a **private** Hugging Face revision (tag `v1.0.1`); publication is
pending the maintainer's decision. A later round of model changes failed its pre-registered evaluation and is not
released.

- `ej.integrity.KNOWN_RELEASES` / `RELEASE_PROVENANCE` list v1.0.1; `scripts/maintainer_pickle.py` checks the encoder's
  sha256 by checkpoint directory (both encoders listed).

- **v1.0.0 withdrawn** before publication: its training pool contained Amazon counterfactual data, licensed CC BY-NC 4.0
  upstream (the mirror used declared CC BY 4.0). README, model card, data card and NOTICE say so; NOTICE lists creator,
  link, licence URI and "modified" per dataset and discloses the open licence questions Q1-Q4.
- `Model.adapt(examples=None, unlabeled=None)` returns an `ej.AdaptedModel` for one workflow: a per-(question id, option
  key) logit offset fitted on the workflow's labelled records (option tilt), with no change to the weights or the runtime.
  Experimental: the same offsets from unlabelled records; `AdaptedModel.observe(batch)` / `predict(batch, observe=True)`.
  The adaptation table reports record-cluster CIs and the add-one count-prior baseline.
- Process hygiene (audit M-6): `ej.load` restores torch's thread count and RNG state after the runtime's import, and no
  longer monkeypatches `transformers` for the process (the e5 revision pin applies inside ej calls only).
  `predict(records, threads=None)` scopes the thread count to the call (`Model.last_threads` reports it).
- No unpickling (audit M-25): after `ej.load`, every runtime module's `torch.load` / `pickle.load` refuses; the
  maintainer-only pickle export moved out of the package to `scripts/maintainer_pickle.py` (opt-in + sha256).
- Bounded memory (audit M-32): `predict(..., chunk_size=64)` encodes records in chunks; the encoder memo is an LRU of
  `memo_max` texts (default 20,000). Chunking and thread count move probabilities by at most about 6e-7.
- Provenance (audit M-14): `scripts/hf_layout.py` records `code_commit` only when it is clean and in a pushed branch, and
  `research_commit`; `scripts/state_key.py` recomputes a state key from a research tree and its pool;
  `ej.integrity.RELEASE_PROVENANCE` corrects v1.0.0's unreachable `code_commit`.
- Benchmark: one latency protocol (`run_bench.py` records threads observed inside the timed calls, the box and the load
  average); ej latency re-measured per record on one thread; rival latencies from another day are marked not comparable.
  Suite builders and sha256 in `benchmark/suites/`.
- Claims withdrawn (round-20 audit): "smallest" (held for the counted bound only), "fastest" (ej was timed batched, the
  rivals per record), "lowest mean ECE" (Jev was lower on each public suite), "1 CPU thread" for latencies measured with 2,
  "sealed / unbiased" for zs_td (it drove model selection), "154 unseen workflows (public sources)" (43 are GLM-synthetic).
- CI: `.github/workflows/tests.yml` runs the tests that need no weights on pull requests and manual dispatch.

## 1.0.0 (2026-10-08) — withdrawn, never published

Packaged but withdrawn before publication (training pool contained CC BY-NC data; see above). The Hugging Face repository
stayed private and no `v1.0.0` tag exists.

- `ej` Python package: `ej.load(path_or_hf_repo, revision=None)` and `Model.predict(records)`; input validation
  (`ej.validate_records`); an example record (`ej.EXAMPLE_RECORD`).
- Pickle-free weights: safetensors tensors plus a JSON skeleton (state skeleton gzipped), decoded with a class allowlist;
  every file's sha256 is checked before decoding; the runtime modules are checked against `RUNTIME_SHA256`.
- Base model `intfloat/e5-small-v2` pinned to revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`.
- `scripts/hf_layout.py` builds the Hugging Face upload directory (never uploads).
- Benchmark code (`benchmark/`) with rival adapters and the method notes; aggregate results of the final benchmark (run 2).
- Model card, data card, NOTICE and licence texts (code Apache-2.0, weights CC BY-SA 4.0).
