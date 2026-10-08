# Final benchmark, run 2: ej 1.0.0 (2026-10-08)

The released model, ej 1.0.0 (11.2 MB), scored once on the sealed test suites with `benchmark/scoring.py`. The suites are
unchanged since the rivals were run (2026-10-07, `benchmark/run_bench.py <suite> <model> --final`, one model process at a
time on a local CPU), so the rival rows are those sealed runs. The release model was chosen on development suites only,
before this run. Files here: one `<suite>.<model>.summary.json` per rival and suite, and `ej-1.0.0.all-suites.json` (all
suites of the ej run). Aggregates only: no per-record predictions, no suite inputs.

Suites: **td** = Typed Decisions test split, seen workflows (300 records / 1,500 questions); **zs_td** = Typed Decisions
test split, workflow `security_incidents`, absent from ej's training data (100 / 500); **zs_massive** = MASSIVE (en) intent
test split with leak-free option sets (1,000 / 1,000); **tickets** / **tickets_ood** = private support-ticket suites
(169 / 507 and 147 / 441). See `benchmark/README.md`.

| model | size | td acc | zs_td acc | zs_massive acc | tickets acc | tickets_ood acc | mean NLL (4) | mean ECE (4) | ms/record (CPU, 1 thread) |
|---|---|---|---|---|---|---|---|---|---|
| **ej 1.0.0** | 11.2 MB | .727 | .490 | .826 | .702 | .642 | .7256 | .062 | 107 (all suites) |
| Jev 1.13.0 (hosted API) | closed | .737 | .742 | .957 | .730 | .626 | .786 | .080 | 323 (network) |
| laya | 421M params | .766 | .766 | .866 | .669 | .617 | .6975 | .169 | 12,666 |
| kev-0.8b | 0.8B | .415 | .520 | .895 | .679 | .608 | .806 | .083 | 8,979 |
| OpenThai-SystemOne | 0.75B | .515 | .632 | .971 | .659 | .617 | .846 | .157 | 7,404 |
| Julia-1 (llama.cpp BF16) | 144M | .733 | .706 | .792 | .513 | .465 | 1.613 | .231 | 3,718 |
| gliclass-edge v3.0 | 33M | .367 | .484 | .545 | .304 | .367 | 2.157 | .372 | 651 |

Mean NLL (4) and mean ECE (4) = means over td, zs_td, zs_massive and tickets. ms/record = the td suite for rivals, all
suites for ej (prediction-time caches enabled). Every cell can be recomputed from the summary files.

Per suite, ej 1.0.0 (NLL / ECE15 / certified automation coverage with its error rate): td .6502 / .047 / .342 (.091);
zs_td 1.1149 / .105 / 0; zs_massive .4758 / .034 / .806 (.079); tickets .6615 / .063 / .186 (.021); tickets_ood .7509 /
.064 / .093 (.053).

## Contamination and fairness flags (details: `benchmark/METHOD.md`)

- **td / zs_td**: laya was trained on the Typed Decisions train split including `security_incidents`, so zs_td is not
  zero-shot for laya; Julia-1 is likely in-domain (validation sets named after all Typed Decisions workflows); Jev's
  training data is undisclosed and its published Typed Decisions results use the same test split (likely seen or tuned).
  For ej, zs_td is a workflow it never saw.
- **zs_massive**: OpenThai-SystemOne trains on MASSIVE intents (in-domain); kev and Jev unknown to zero-shot.
- **tickets / tickets_ood**: ej was trained on the training split of the same private ticket corpus (in-domain);
  tickets_ood is out-of-distribution tickets. Rivals see both zero-shot. Do not read ej's tickets results as a general win.
- **Jev**: probabilities arrive rounded to 2 decimals (hurts its NLL when the gold option gets 0.00) and are not
  deterministic (one cached draw per request).
- **Latency**: local CPU, 1 thread, sometimes on a busy machine (upper bounds); Jev is a network round trip.

## Reading

ej is the smallest and fastest model in the table (11.2 MB, ~0.1 s per record on one CPU thread vs 3.7-13 s for the
144M-0.8B local rivals) and has the lowest mean ECE; it is second on mean NLL behind laya. On the unseen workflow it is
level with gliclass-edge (.490 vs .484) and far ahead of it on every other suite. It trails the 144M-0.8B rivals and the
hosted Jev on td / zs_td / zs_massive accuracy. zs_td is one workflow of 100 records (500 questions), and two fits of
identical code differ by about .05 there, so small differences on that suite are within fit-to-fit variation. Published
reference points on the 400 Typed Decisions test cases: laya 0.766, Jev 0.727; reproduced here as laya .766 and Jev
.737 / .742.
