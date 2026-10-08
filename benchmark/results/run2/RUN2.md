# Final benchmark, run 2: ej 1.0.0 (2026-10-08) — v1.0.0 (withdrawn: training pool contained CC BY-NC data)

**Withdrawn.** ej 1.0.0 was never published: its training pool contained Amazon counterfactual data, licensed CC BY-NC 4.0
upstream (`NOTICE`). This page keeps its run-2 numbers for the record, with the corrections of the round-20 audit below.
The v1.1.0 table replaces it when the v1.1.0 evaluation has run ({{V110_BENCH_TABLE}}).

ej 1.0.0 was scored once on the final test suites with `benchmark/scoring.py`. The suites are unchanged since the rivals
were run (2026-10-07, `benchmark/run_bench.py <suite> <model> --final`, one model process at a time on a local CPU), so
the rival rows are those runs. Files here: one `<suite>.<model>.summary.json` per rival and suite, and
`ej-1.0.0.all-suites.json` (all suites of the ej run). Aggregates only: no per-record predictions, no suite inputs.

Suites: **td** = Typed Decisions test split, seen workflows (300 records / 1,500 questions); **zs_td** = Typed Decisions
test split, workflow `security_incidents`, absent from ej's training data but used for model selection on its development
records (100 / 500); **zs_massive** = MASSIVE (en) intent test split with leak-free option sets (1,000 / 1,000);
**tickets** / **tickets_ood** = private support-ticket suites (169 / 507 and 147 / 441). See `benchmark/README.md`.

| model | size as published | td micro acc | zs_td micro acc | zs_massive micro acc | tickets micro acc | tickets_ood micro acc | mean NLL (4) | mean ECE (4) |
|---|---|---|---|---|---|---|---|---|
| ej 1.0.0 (withdrawn) | 11.2 MiB counted (on disk 36.1 MB; download about 170.5 MB) | .727 | .490 | .826 | .702 | .642 | .7256 | .062 |
| Jev 1.13.0 (hosted API) | closed | .737 | .742 | .957 | .730 | .626 | .786 | .080 |
| laya | 421M params | .766 | .766 | .866 | .669 | .617 | .6975 | .169 |
| kev-0.8b | 0.8B | .415 | .520 | .895 | .679 | .608 | .806 | .083 |
| OpenThai-SystemOne | 0.75B | .515 | .632 | .971 | .659 | .617 | .846 | .157 |
| Julia-1 (llama.cpp BF16) | 144M | .733 | .706 | .792 | .513 | .465 | 1.613 | .231 |
| gliclass-edge v3.0 | 33M | .367 | .484 | .545 | .304 | .367 | 2.157 | .372 |

Mean NLL (4) and mean ECE (4) = means over td, zs_td, zs_massive and tickets. Every cell can be recomputed from the
summary files. **No CI was computed for any cell**; question-level intervals are at least about ±.02-.04 on td and zs_td,
so orderings inside that range are not claims.

Per suite, ej 1.0.0 (NLL / ECE15 / certified automation coverage with its error rate, this suite only): td .6502 / .047 /
.342 (.091); zs_td 1.1149 / .105 / 0; zs_massive .4758 / .034 / .806 (.079); tickets .6615 / .063 / .186 (.021);
tickets_ood .7509 / .064 / .093 (.053).

## Corrections (round-20 audit)

- **Latency column removed.** The published ej figure (107 ms) was the unweighted mean of whole-suite batched calls run
  with 2 torch threads (the runtime set 2 at import), while the rivals were timed one record per call with 1 thread on
  the td suite. The rival figures (per record, td, 1 thread: Jev 323 (network), laya 12,666, kev-0.8b 8,979, OpenThai
  7,404, Julia-1 3,718, gliclass-edge 651 ms) were measured on another day under unrecorded load and are **not
  comparable** with ej's re-measured per-record latency (README "Latency"); they are not ranked against it.
- **Size.** "11.2 MB" was a counted bit-level bound in MiB, not a file size; by download, ej 1.0.0 (about 170.5 MB with
  the full base model) was larger than gliclass-edge (about 134.4 MB). No size ranking is claimed.
- **Calibration.** Over the three public suites (td, zs_td, zs_massive) Jev's ECE15 is lower than ej's on each
  (.0505 vs .0620 mean); ej's lower 4-suite mean comes from the tickets suite it was trained for. No calibration ranking
  is claimed.
- **zs_td** drove model selection (the release was kept on its development accuracy), so it is neither unbiased nor
  unseen-workflow evidence (`docs/MODEL_CARD.md` §4).

## Contamination and fairness flags (details: `benchmark/METHOD.md`)

- **td / zs_td**: laya was trained on the Typed Decisions train split including `security_incidents`, so zs_td is not
  zero-shot for laya; Julia-1 is likely in-domain (validation sets named after all Typed Decisions workflows); Jev's
  training data is undisclosed and its published Typed Decisions results use the same test split (likely seen or tuned).
  For ej, zs_td is a workflow it never trained on but was selected on.
- **zs_massive**: OpenThai-SystemOne trains on MASSIVE intents (in-domain); kev and Jev unknown to zero-shot. About 22% of
  MASSIVE's intent names overlap CLINC150 / Banking77 option texts in ej's pool.
- **tickets / tickets_ood**: ej was trained on the training split of the same private ticket corpus (in-domain);
  tickets_ood holds tickets in held-out writing styles (a style shift). Rivals see both zero-shot. Do not read ej's
  tickets results as a general win.
- **Jev**: probabilities arrive rounded to 2 decimals (hurts its NLL when the gold option gets 0.00) and are not
  deterministic (one cached draw per request; re-sent requests moved values by up to .04, no spread measured).

## Reading

ej 1.0.0 was second on mean NLL behind laya. On zs_td its micro accuracy was close to gliclass-edge's (.490 vs .484, within
the question-level interval). It trailed the 144M-0.8B rivals and the hosted Jev on td / zs_td / zs_massive micro accuracy.
zs_td is one workflow of 100 records (500 questions); two fits of identical code that differ only in their seed differ by
about .03 there (SD of a difference). Published reference points on the 400 Typed Decisions test cases: laya 0.766, Jev
0.727 (as quoted by the Typed Decisions leaderboard; not re-checked here); measured here: laya .766, Jev .737 / .742 on
td / zs_td, from one non-deterministic Jev draw (+.010 / +.015 against the quoted value).
