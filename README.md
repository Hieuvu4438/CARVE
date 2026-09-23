# CARVE-simple + HONE

**Propose-then-verify argument extraction for scientific event extraction.**
Evaluated on SciEvent (Dong et al., EMNLP 2025).

- **Stage 1, CARVE-simple** (the *proposer*): a DeBERTa-v3-large word-level span tagger with a single 13-type BIO head and a window event-type head.
- **Stage 2, HONE** (*Hard-negative Out-of-fold Neural vErification*): a cross-encoder verifier that keeps, rejects or relabels every candidate span the proposer produces. It is trained on out-of-fold candidates, so it learns from the proposer's real mistakes.

## Results (SciEvent test, official scorer)

The decoding rule was frozen on dev before test was touched. The released system uses a single proposer seed and a single verifier seed (42 / 42).

**Trigger ROUGE-L** (paper Table 3):

| Method | P | R | **F1** |
|---|---|---|---|
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT 5-shot (best in paper) | 73.70 | 78.82 | 75.08 |
| CARVE-simple (no verifier, 3 seeds) | 84.77 ± 0.43 | 76.15 ± 1.21 | 77.22 ± 0.81 |
| **CARVE-simple + HONE** | **85.13** | 76.78 | **77.82** |

**Argument extraction, IoU > 0.5** (paper Table 4):

| Method | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE (best in paper) | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT 5-shot | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| CARVE-simple (no verifier, 3 seeds) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |
| CARVE-simple (no verifier, seed 42) | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| **CARVE-simple + HONE** | **70.18** | 52.53 | **60.09** | **61.15** | 45.78 | **52.36** |

**Paired window bootstrap** (5,000 resamples of the 163 test windows):

| comparison | Arg-C Δ [95 % CI] | Arg-I Δ [95 % CI] |
|---|---|---|
| CARVE-simple + HONE vs OneIE (published aggregate) | +10.75, system CI [47.28, 57.18] | +6.52, system CI [55.12, 64.72] |
| CARVE-simple + HONE vs CARVE-simple, same seed | +1.66 [−0.14, +3.44] | +1.74 [−0.65, +4.03] |

Reading these results:

- The margin over the paper's best baseline is robust: the lower bound of the system's own confidence interval exceeds OneIE on both metrics.
- The verifier's gain over its own proposer (+1.66 Arg-C) is **not** statistically established on one seed.
- The verifier changes the precision/recall balance: +6.5 precision for −1.5 recall on Arg-C.
- ROUGE-L comes from the proposer (the trigger and the ⟨Agent, Action, Object⟩ tuple are not verified), so it is not a contribution of HONE.
- Full tables, per-seed numbers, all four matching modes, the dev results and every caveat are in [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md). A Vietnamese paper-style write-up is in [`docs/PAPER_VI.md`](docs/PAPER_VI.md).

## Method

```
window ──► CARVE-simple ──► every argmax span of a scored role (no threshold) + 17 evidence features
                                        │
                                        ▼
         "event type: <T> . w1 … <a> span </a> … wn"  +  proposer features
                                        │
                        HONE verifier: DeBERTa-v3-large cross-encoder
                                        │
                                        ▼
                  {reject, Context, Method, Results, …}  per candidate
                                        │
                                        ▼
   keep score ≥ θ · role = α-mix(verifier, proposer) · joint event type (λ) · no overlaps
```

1. **Proposer (CARVE-simple).**
   - DeBERTa-v3-large with first-subword pooling.
   - One BIO head over 13 span types: the 9 scored roles plus Agent, Action, PrimaryObject and SecondaryObject.
   - An attention-pooled 4-way event-type head.
   - Trained 30 epochs; the checkpoint is selected on dev under calibrated decoding.
   - Why it is "simple": the original CARVE had two further design decisions, separate heads for the two span groups and conditioning roles on the predicted event type. Single-factor ablations showed both to be inert, so both were removed.
2. **Out-of-fold candidates.**
   - The proposer memorises its training windows: 97.5 % of its in-sample candidates are correct, against about 38 % out of sample. A verifier trained on in-sample candidates learns to keep everything.
   - The train split is therefore cut into 5 event-type-stratified folds. Each fold is decoded by a proposer trained on the other four.
   - Result: 5.06 candidates per window, 38.0 % positive, which matches dev.
3. **Verifier (HONE).**
   - Entity-marker cross-encoder: `[CLS; h(<a>); h(</a>); MLP(features)] → 10 classes`.
   - Candidate labels follow the metric: the role of the gold span with IoU > 0.5, else `reject`.
   - Trained 5 epochs; the epoch is selected on dev.
4. **Decoding.**
   - Keep score = 1 − p(reject).
   - Role = α-mix of the verifier's role distribution and the proposer's role mass.
   - Window type = argmax_t log p(t) + λ Σ log P(role | t), with P(role | t) estimated from train gold.
   - Spans are chosen greedily with no token overlap.
   - Frozen on dev: θ = 0.20, min_len = 1, α = 0.25, λ = 0.5.

## Installation

```bash
git clone https://github.com/Hieuvu4438/CARVE.git
cd CARVE
pip install -e .
bash scripts/setup_data.sh          # clones SciEvent into third_party/ and runs its own preprocessing
python3 tests/test_contract.py      # data contract + official-scorer oracle (must print 100.00)
```

The benchmark is a read-only dependency. The official scorer
(`third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py`) is imported, never
reimplemented. `SCIEVENT_ROOT` can point at an existing checkout instead.

## Reproduction

```bash
bash scripts/reproduce.sh
```

`scripts/reproduce.sh` runs every step in order:

| step | command | output |
|---|---|---|
| proposers (seeds 42, 13, 101) | `python3 -m carve.train -c configs/proposer.json --set run_name=proposer_s42 seed=42` | `runs/proposer_s*/` |
| out-of-fold candidates | `python3 scripts/propose.py oof --fold K --seed 42` (K = 0..4) | `data/cands/oof_k*_s42.jsonl` |
| dev/test candidates | `python3 scripts/propose.py eval --seed 42` | `data/cands/{dev,test}_s42.jsonl` |
| verifier | `python3 scripts/train_verifier.py --prop_seed 42 --seed 42` | `runs/verifier_s42/` |
| freeze rules (dev only) | `scripts/proposer_baseline.py freeze`, `scripts/freeze_and_test.py freeze` | `assets/*.json` |
| single test evaluation | `scripts/proposer_baseline.py test`, `scripts/freeze_and_test.py test` | `preds/` |
| paired bootstrap | `python3 scripts/bootstrap_ci.py` | stdout |

`assets/proposer_rule.json` and `assets/decoding_rule.json` are the frozen rules of the reported run, with their timestamps.

**What reproduces exactly, and what does not:**

- **Checkpoint inference is exact.** Run on the reported checkpoints, the released code reproduces the reported candidate files and test predictions byte for byte, and re-freezing gives the same rule.
- **Retraining is not bit-exact.** GPU training is nondeterministic: two runs of the same code already differ after one epoch.
- **Expect seed-level variation on retraining.** The proposer's measured seed spread is ±0.19 test Arg-C. The verifier was trained with one seed only, so its spread has not been measured.

About 2 GPU-hours on one 48 GB GPU.

## Repository layout

```
carve/
  model.py        CARVE-simple tagger (stage 1)
  train.py        proposer training + dev checkpoint selection
  decode.py       proposer posteriors + threshold decoder (baseline)
  candidates.py   candidate extraction, normalisation, metric-faithful labels
  verifier.py     HONE cross-encoder (stage 2)
  select.py       frozen decoding rule (stage 3)
  data.py         benchmark loading, label inventories
  evaluate.py     wrapper around the official scorer
  paths.py        benchmark location
scripts/          setup_data, propose, train_verifier, freeze_and_test,
                  proposer_baseline, bootstrap_ci, reproduce
configs/          proposer.json
assets/           frozen decoding rules of the reported run
tests/            contract tests (CPU)
docs/             PAPER_VI.md (write-up), PAPER_NOTES.md (exact method + all numbers)
```

## Limitations

- **One seed.** The released system has a single proposer seed and a single verifier seed. The verifier's gain over its own proposer is positive on test but not significant, and it is ≈ 0 on dev (49.16 vs 49.19).
- **Test-look history.** This configuration was chosen after several systems had been evaluated on test during development. None of those looks selected it by its test score, but they are disclosed in `docs/PAPER_NOTES.md`.
- **Not the top-scoring configuration measured.** Other configurations from development scored higher on test and are archived with full records in the [SciEvent research repository](https://github.com/Hieuvu4438/SciEvent) (`method/HONE`):
  - the same verifier on the original two-head CARVE proposer: 53.35 Arg-C, one seed;
  - three pooled proposer seeds: 53.06 ± 1.04 Arg-C, 63.05 Arg-I.
- **The official split shares abstracts across splits.** It is window-level: 97.3 % of test abstracts also have windows in train. All published baselines share this property.
- **No `sent_id` input.** The window-ID suffix (`-0`, `-1`, …) predicts the gold event type 94–99 % of the time. It is annotation metadata, and no model input reads it.

## History

This repository previously contained the original CARVE: two BIO heads, event-type conditioning and a calibrated threshold, test Arg-C 50.48 ± 1.15. That code, its ablations and its write-ups are archived in the SciEvent research repository under `method/CARVE-full/`, and remain in this repository's git history.

## License

See [LICENSE](LICENSE). The SciEvent benchmark is subject to its own license.
