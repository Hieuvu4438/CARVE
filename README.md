# CLAVE: Cross-Fitted Clause-Level Argument Verification for Scientific Event Extraction

**CLAVE** is a propose-then-verify system for argument extraction on SciEvent (Dong et al., EMNLP 2025).

- **Stage 1, the proposer (CARVE-simple).** A DeBERTa-v3-large word-level span tagger with a single 13-type BIO head and a window event-type head. It proposes every clause-level argument span it decodes, with no threshold.
- **Stage 2, the verifier (HONE, *Hard-negative Out-of-fold Neural vErification*).** A cross-encoder that keeps, rejects or relabels each candidate span. It is trained on **cross-fitted** (out-of-fold) candidates, so it learns from the proposer's real mistakes.

The name spells out the method: **cl**ause-level **a**rgument **ve**rification, with the verifier trained on cross-fitted candidates.

## Results (SciEvent test, official scorer)

The current three-seed result pairs proposer and verifier seeds (42/42, 13/13, 101/101) under one rule selected on dev and frozen before its registered test evaluation. The original one-seed release is reported separately.

**Trigger ROUGE-L** (paper Table 3):

| Method | P | R | **F1** |
|---|---|---|---|
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT 5-shot (best in paper) | 73.70 | 78.82 | 75.08 |
| CARVE-simple (no verifier, 3 seeds) | 84.77 ± 0.43 | 76.15 ± 1.21 | 77.22 ± 0.81 |
| CLAVE, original 1-seed release | 85.13 | 76.78 | 77.82 |
| **CLAVE, 3 seeds** | **84.77 ± 0.43** | **76.15 ± 1.21** | **77.22 ± 0.81** |

**Argument extraction, IoU > 0.5** (paper Table 4):

| Method | ArgI-P | ArgI-R | **ArgI-F1** | ArgC-P | ArgC-R | **ArgC-F1** |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE (best in paper) | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT 5-shot | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| CARVE-simple (no verifier, 3 seeds) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |
| CARVE-simple (no verifier, seed 42) | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| CLAVE, original 1-seed release | 70.18 | 52.53 | 60.09 | 61.15 | 45.78 | 52.36 |
| **CLAVE, 3 seeds** | **69.93 ± 6.63** | **53.97 ± 6.23** | **60.49 ± 1.25** | **60.77 ± 7.27** | **46.72 ± 4.01** | **52.45 ± 0.73** |

**Paired window bootstrap** (5,000 resamples of the 163 test windows):

| comparison | Arg-C Δ [95 % CI] | Arg-I Δ [95 % CI] |
|---|---|---|
| CLAVE, 3 seeds vs OneIE (published aggregate) | +10.84, system CI [47.87, 57.08] | +6.92, system CI [56.22, 64.61] |
| CLAVE, 3 seeds vs CARVE-simple, paired windows | +1.72 [+0.24, +3.30] | +1.90 [+0.21, +3.72] |
| Original 1-seed CLAVE vs CARVE-simple, same seed | +1.66 [−0.14, +3.44] | +1.74 [−0.65, +4.03] |

Reading these results:

- The lower bound of CLAVE's own bootstrap interval exceeds OneIE's published point score on both argument metrics; per-window OneIE predictions are unavailable for a paired comparison.
- The three-seed paired test contrast is +1.72 Arg-C with a positive bootstrap CI. Its pre-registered dev gain is +0.33, below the +0.5 threshold for supporting hypothesis G1.
- The verifier changes the precision/recall balance: +6.52 precision for −0.94 recall on Arg-C, averaged over the three paired runs.
- ROUGE-L comes from the proposer (the trigger and the ⟨Agent, Action, Object⟩ tuple are not verified), so it is not a contribution of HONE.
- The three runs reach similar F1 at different precision/recall operating points. The seed-13 verifier was selected at epoch 1 and keeps more spans (Arg-C P/R 52.4 / 51.2, against about 65 / 44 for the other two runs). So the P and R standard deviations are large while F1 is stable (±0.73).
- Full generated tables, ablations and descriptive analysis are in [`experiment/tables/`](experiment/tables/). The protocol and test-look log are in [`experiment/`](experiment/); [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md) covers the new result and release history, with a Vietnamese write-up in [`docs/PAPER_VI.md`](docs/PAPER_VI.md).

## Ablations (3 seeds, each variant with its own dev-frozen rule)

| variant | dev Arg-C | test Arg-C | Δ test vs CLAVE [95 % CI] |
|---|---:|---:|---:|
| **CLAVE** | 48.43 ± 0.47 | **52.45 ± 0.73** | — |
| − verifier (CARVE-simple threshold) | 48.10 ± 0.96 | 50.73 ± 0.19 | −1.72 [−3.30, −0.24] |
| − cross-fitting (verifier trained in-sample) | 46.80 ± 0.79 | 48.97 ± 0.61 | −3.48 [−4.92, −2.10] |
| − proposer features (text-only verifier) | 49.08 ± 1.45 | 51.46 ± 1.22 | −0.99 [−2.03, 0.00] |
| verifier roles only (α = 1) | 47.36 ± 1.08 | 50.00 ± 1.58 | −2.46 [−4.35, −0.56] |
| proposer roles only (α = 0) | 48.06 ± 0.59 | 52.16 ± 0.31 | −0.29 [−0.73, +0.08] |
| − joint event type (λ = 0) | 48.35 ± 0.34 | 52.30 ± 0.82 | −0.15 [−0.48, 0.00] |
| proposer decoded by argmax (no calibration, no verifier) | 38.32 ± 0.25 | 39.67 ± 0.76 | −12.78 (vs CLAVE) |

**Proposer design, 2×2 on test (Arg-C):**

| | with type conditioning | without type conditioning |
|---|---:|---:|
| two heads | 50.48 ± 1.15 | 49.82 ± 0.62 |
| one merged head | 50.76 ± 0.39 | **50.73 ± 0.19 (CARVE-simple)** |

No cell differs from CARVE-simple; every paired CI covers zero.

**Pre-registered hypotheses** (support threshold: ≥ +0.5 dev Arg-C):

| hypothesis | dev result |
|---|---|
| G1: verifier helps (+0.33 dev) | rejected on dev, but the test contrast is positive (+1.72, CI above 0) |
| G2: cross-fitting | supported |
| G3: proposer features | rejected (text-only is better on dev) |
| G4: role mixing | supported |
| G5: joint event type | rejected |

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
   - The proposer memorises its training windows: 98.1–98.6 % of its in-sample candidates are correct across three seeds, against 37.0–38.5 % out of fold. A verifier trained on in-sample candidates sees a very different label distribution.
   - The train split is therefore cut into 5 event-type-stratified folds. Each fold is decoded by a proposer trained on the other four.
   - Across the three seeds: 5.01–5.06 out-of-fold candidates per window, 37.0–38.5 % positive; dev has 5.01–5.16 candidates per window.
3. **Verifier (HONE).**
   - Entity-marker cross-encoder: `[CLS; h(<a>); h(</a>); MLP(features)] → 10 classes`.
   - Candidate labels follow the metric: the role of the gold span with IoU > 0.5, else `reject`.
   - Trained 5 epochs; the epoch is selected on dev.
4. **Decoding.**
   - Keep score = 1 − p(reject).
   - Role = α-mix of the verifier's role distribution and the proposer's role mass.
   - Window type = argmax_t log p(t) + λ Σ log P(role | t), with P(role | t) estimated from train gold.
   - Spans are chosen greedily with no token overlap.
   - Three-seed rule frozen on dev: θ = 0.30, min_len = 3, NMS `any`, α = 0.50, λ = 0.5 (`experiment/assets/decoding_rule_3seed.json`). The original release used θ = 0.20, min_len = 1, α = 0.25, λ = 0.5.

## Installation

```bash
git clone https://github.com/Hieuvu4438/CLAVE.git
cd CLAVE
pip install -e .
bash scripts/setup_data.sh          # clones SciEvent into third_party/ and runs its own preprocessing
python3 tests/test_contract.py      # data contract + official-scorer oracle (must print 100.00)
```

The benchmark is a read-only dependency. The official scorer
(`third_party/SciEvent/baselines/ONEIE/EM_overlap_eval.py`) is imported, never
reimplemented. `SCIEVENT_ROOT` can point at an existing checkout instead.

## Reproduction

```bash
bash scripts/reproduce.sh 3-seed 1    # start the three-seed pipeline; review each STOP before the next phase
CLAVE_RUN_P2=1 bash scripts/reproduce.sh 3-seed 7  # optional approved T-P2 before test analysis
bash scripts/reproduce.sh released    # original one-seed pipeline
```

The three-seed experiment runs phases 1–7 separately, with a stop and audit after each phase. Set `CLAVE_RUN_P2=1` only when the optional six-run proposer ablation is authorized; it runs after T-P1 and before the descriptive test analysis. `p2-train` and `p2-test` are also available separately for that approved ablation. Outputs and frozen rules are under `experiment/`; see [`experiment/EXPERIMENT_PLAN.md`](experiment/EXPERIMENT_PLAN.md). Existing test looks cannot be replayed in place. Use a clean checkout and follow the phase order for a full retraining reproduction.

The `released` mode runs the original steps in order:

| step | command | output |
|---|---|---|
| proposers (seeds 42, 13, 101) | `python3 -m clave.train -c configs/proposer.json --set run_name=proposer_s42 seed=42` | `runs/proposer_s*/` |
| out-of-fold candidates | `python3 scripts/propose.py oof --fold K --seed 42` (K = 0..4) | `data/cands/oof_k*_s42.jsonl` |
| dev/test candidates | `python3 scripts/propose.py eval --seed 42` | `data/cands/{dev,test}_s42.jsonl` |
| verifier | `python3 scripts/train_verifier.py --prop_seed 42 --seed 42` | `runs/verifier_s42/` |
| freeze rules (dev only) | `scripts/proposer_baseline.py freeze`, `scripts/freeze_and_test.py freeze` | `assets/*.json` |
| single test evaluation | `scripts/proposer_baseline.py test`, `scripts/freeze_and_test.py test` | `preds/` |
| paired bootstrap | `python3 scripts/bootstrap_ci.py` | stdout |

`assets/proposer_rule.json` and `assets/decoding_rule.json` are the frozen rules of the original release. The three-seed CLAVE rule is `experiment/assets/decoding_rule_3seed.json`.

**What reproduces exactly, and what does not:**

- **Checkpoint inference is exact.** Run on the reported checkpoints, the released code reproduces the reported candidate files and test predictions byte for byte, and re-freezing gives the same rule.
- **Retraining is not bit-exact.** GPU training is nondeterministic: two runs of the same code already differ after one epoch.
- **Expect seed-level variation on retraining.** The observed three-seed test Arg-C spread is ±0.73 for CLAVE and ±0.19 for CARVE-simple.

The original release pipeline takes about 2 GPU-hours on one 48 GB GPU; the three-seed extension and optional T-P2 add more training.

## Repository layout

```
clave/
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
experiment/       registered three-seed protocol, phase reports, frozen rules, results and tables
```

## Limitations

- **Three seeds remain a small sample.** The registered three-seed dev improvement over CARVE-simple is only +0.33 Arg-C, below the planned +0.5 threshold, despite a positive paired test contrast. The original release used one seed.
- **Two retained components are not supported.** The proposer features (G3) and the joint event-type decision λ (G5) failed their pre-registered dev test. They stay in the frozen system because removing them does not help, but they are not claimed as contributions. The supported contributions are cross-fitting (G2), role mixing (G4) and calibrated proposer decoding.
- **Test-look history.** The original release configuration was chosen after several systems had been evaluated on test during development. That history and the newer registered looks are disclosed in `docs/PAPER_NOTES.md` and `experiment/TEST_LOG.md`.
- **Not the top-scoring configuration measured.** Other configurations from development scored higher on test and are archived with full records in the [SciEvent research repository](https://github.com/Hieuvu4438/SciEvent) (`method/HONE`):
  - the same verifier on the original two-head CARVE proposer: 53.35 Arg-C, one seed;
  - three pooled proposer seeds: 53.06 ± 1.04 Arg-C, 63.05 Arg-I.
- **The official split shares abstracts across splits.** It is window-level: 97.3 % of test abstracts also have windows in train. All published baselines share this property.
- **No `sent_id` input.** The window-ID suffix (`-0`, `-1`, …) predicts the gold event type 94–99 % of the time. It is annotation metadata, and no model input reads it.

## History

This repository was previously named **CARVE** and contained the original CARVE: two BIO heads, event-type conditioning and a calibrated threshold, test Arg-C 50.48 ± 1.15.
- CLAVE's proposer, CARVE-simple, is that model with the two inert components removed.
- The old code, its ablations and its write-ups are archived in the SciEvent research repository under `method/CARVE-full/`, and remain in this repository's git history.
- GitHub redirects the old URL `github.com/Hieuvu4438/CARVE` here.

## License

See [LICENSE](LICENSE). The SciEvent benchmark is subject to its own license.
