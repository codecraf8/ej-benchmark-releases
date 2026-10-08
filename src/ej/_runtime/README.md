# ej/_runtime: the prediction code

These 39 modules are the prediction code of ej. They are derived from the research repository's `edge/` directory at
commit 46d0731 (named in `RUNTIME_SHA256`): exactly the modules that load when a prediction runs (found by running a
prediction and listing the imported modules). For this repository their **code is unchanged**; only module docstrings,
some function docstrings and comments were rewritten (internal notes removed), and the research-machine path defaults
of a few environment variables were replaced by neutral ones under `~/.cache`. A syntax-tree comparison with the source
(docstrings ignored) shows no other difference, and predictions on the reference records are identical (max |Δp| = 0.0).

- **Do not edit these files**: `ej.integrity.verify_runtime()` checks every file against `RUNTIME_SHA256` on each
  `ej.load()` and refuses to load on any difference, and a weights release records the sha256 of the `RUNTIME_SHA256` it
  was packaged for.
- The modules import each other by **bare name** (`import student_lb`, ...). `ej.load()` puts this directory at the front of
  `sys.path`; a module of your own named like one of these files (e.g. `student.py`) would clash, and `ej.load()` refuses
  to continue when it detects that.
- At prediction time ej sets `EDGE_CKPT`, `EDGE_CACHE` and `HF_HOME` before the first import, so the defaults in the code
  are not used.
- Importing the runtime sets `torch.manual_seed(0)` and `torch.set_num_threads(2)` for the process.
- Fit-time code paths (training, teachers, cross-fitting) are present but never run by ej.
