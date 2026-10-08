# Changelog

## 1.0.0 (2026-10-08)

First public release.

- `ej` Python package: `ej.load(path_or_hf_repo, revision=None)` and `Model.predict(records)`; input validation
  (`ej.validate_records`); an example record (`ej.EXAMPLE_RECORD`).
- Pickle-free weights: safetensors tensors plus a JSON skeleton (state skeleton gzipped), decoded with a class allowlist;
  every file's sha256 is checked before decoding; the runtime modules are checked against `RUNTIME_SHA256`.
- Base model `intfloat/e5-small-v2` pinned to revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`.
- `scripts/hf_layout.py` builds the Hugging Face upload directory (never uploads).
- Benchmark code (`benchmark/`) with rival adapters and the method notes; aggregate results of the final benchmark (run 2).
- Model card, data card, NOTICE and licence texts (code Apache-2.0, weights CC BY-SA 4.0).
