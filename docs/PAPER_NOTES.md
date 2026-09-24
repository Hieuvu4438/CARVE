# CLAVE — paper notes

**Title:** *CLAVE: Cross-Fitted Clause-Level Argument Verification for Scientific Event Extraction.*

CLAVE = the CARVE-simple proposer (stage 1) + the HONE verifier (stage 2). The component names are kept because the development record (SciEvent repository, `method/HONE`) uses them.

Exact method, protocol, and every number with its provenance. Numbers are on
the official SciEvent split (1,278 / 158 / 163 windows), scored with the
benchmark's own `EM_overlap_eval.py` (imported verbatim). "Seed 42" means
proposer seed 42 and verifier seed 42 unless stated.

**Current main result:** the registered three-seed extension is in §10 and
`experiment/tables/T1.md`–`T8.md`. Sections 0–9 retain the original one-seed
release and its development history.

---

## 0. Claims

| claim | status | evidence |
|---|---|---|
| CLAVE beats the best published baseline (OneIE) on Arg-C and Arg-I IoU | **supported** | test Arg-C 52.36 vs 41.61; the system's own 95 % CI [47.28, 57.18] lies above it (§4.6) |
| … on trigger ROUGE-L vs GPT 5-shot | **point estimate only** | 77.82 vs 75.08; no per-window GPT predictions, so no paired test; ROUGE-L comes from the proposer |
| HONE improves over its own proposer (CARVE-simple, same seed) | **not established** | Arg-C +1.66, CI [−0.14, +3.44]; dev ≈ 0 (49.16 vs 49.19) |
| CARVE-simple ≡ original CARVE | **supported (no difference)** | test Δ +0.25, CI [−1.66, +2.12]; lower seed variance (§6.1) |
| out-of-fold candidates are necessary | **supported (original-CARVE proposer)** | in-sample-trained verifier −4.3 dev (§5) |

## 1. Task and evaluation contract

- A window is one annotated abstract segment with exactly one event: 4 event types, 9 scored argument roles, plus an ⟨Agent, Action, Object⟩ tuple.
- **Arg-I / Arg-C:**
  - one-to-one greedy matching at IoU > 0.5 in whitespace-token space;
  - the event type must match;
  - Arg-C additionally requires the right role;
  - trigger-insensitive;
  - Agent, PrimaryObject and SecondaryObject are excluded from scoring.
- **Trigger ROUGE-L:** ROUGE-L between the predicted and gold ⟨Agent, Action, Object⟩ strings.
- **Scorer integrity:** the official scorer is imported, never reimplemented. Gold routed through our exporter scores 100.00 on every metric, on dev and on test (`tests/test_contract.py`).

## 2. Method

### 2.1 Stage 1 — CARVE-simple proposer (`clave/model.py`, `clave/train.py`, `configs/proposer.json`)

| component | setting |
|---|---|
| encoder | `microsoft/deberta-v3-large`, fp32 master weights, bf16 autocast |
| word states | first sub-token of each whitespace word, dropout 0.1 |
| span head | Linear(H,H) → GELU → Dropout → Linear(H, 27): BIO over 13 types (9 roles + Agent, Action, PrimaryObject, SecondaryObject) |
| type head | attention pooling → Linear(H, 4) |
| BIO conflicts | shorter span wins (`spans_to_bio`); merged round-trip recovers 98.05 % of gold role spans |
| loss | CE(span BIO) + 0.5 · CE(event type) |
| optimiser | AdamW, lr 1e-5 encoder / 1e-4 heads, weight decay 0.01, 10 % linear warmup, grad clip 1.0 |
| schedule | 30 epochs, batch 8, max 640 sub-tokens |
| checkpoint selection | dev Arg-C after thresholding, τ swept over {0.3 … 0.95} each epoch, min_len 3 |

- **Removed from the original CARVE:**
  - a separate AAO head;
  - type conditioning of the role head;
  - optional CRF, span-level role head and LLRD (all rejected earlier; §5).
- **Two harmless leftovers, kept deliberately:** an unused zero-initialised embedding and one unused random draw per batch. They make training follow the reported run's random stream; see the comment in `clave/model.py`.

### 2.2 Candidates (`scripts/propose.py`, `clave/candidates.py`)

**Extraction:**
- The proposer decodes the merged posterior by argmax.
- Every span of a scored role becomes a candidate, with no threshold.
- AAO spans and the trigger come from the same argmax.

**Candidate features (17):**
- role mass for each of the 9 roles (mean P(B-r) + P(I-r) over the span);
- P(O);
- mean confidence and max confidence;
- seed agreement (always 1/3, since one seed is used);
- log length;
- relative start and relative end;
- a part count (always 0; it is a leftover of the rejected composition variant, kept so the input layout matches the reported run).

**Out-of-fold training candidates:**
- 5 window-level folds, stratified by event type (`FOLD_SEED = 0`).
- Per fold, a proposer is trained on the other 4 folds with the same recipe, its epoch selected on the real dev split, and it decodes the held-out fold.
- The fold checkpoint is then deleted.

| candidates (proposer seed 42) | count | per window | positive |
|---|---|---|---|
| train, out-of-fold | 6,468 | 5.06 | 38.0 % |
| dev, full-train proposer | 792 | 5.01 | 42.2 % |
| test, full-train proposer | 832 | 5.10 | — (not inspected) |

- On matched dev candidates the proposer's own role is right 80.2 % of the time.
- Oracle verifier over the dev candidates (perfect keep and role, same decoder): **72.20** Arg-C. That is the headroom above the 49.16 achieved.

### 2.3 Stage 2 — HONE verifier (`clave/verifier.py`, `scripts/train_verifier.py`)

| component | setting |
|---|---|
| input | `event type: <predicted type> .` + window with `<a> … </a>` around the candidate (two added special tokens), max 320 sub-tokens |
| encoder | `microsoft/deberta-v3-large`, fp32 master weights, bf16 autocast, gradient checkpointing |
| head | [h_CLS; h_<a>; h_</a>; GELU(Linear(17 → 64))] → Linear → GELU → Dropout 0.1 → Linear(10) |
| target | role of the max-IoU gold span if IoU > 0.5, else `reject` |
| loss | cross-entropy, no label smoothing |
| optimiser | AdamW, lr 1e-5 encoder / 1e-4 head, weight decay 0.01, 10 % warmup, grad clip 1.0 |
| schedule | 5 epochs, batch 8 × 2 accumulation steps, seed 42 |
| epoch selection | dev Arg-C under a verifier-only rule tuned per epoch |

Training curve on dev, verifier-only rule:

| epoch | 1 | 2 | 3 | 4 | **5** |
|---|---|---|---|---|---|
| dev Arg-C | 37.78 | 46.49 | 46.47 | 45.81 | **46.69** |

### 2.4 Stage 3 — decoding (`clave/select.py`, `scripts/freeze_and_test.py`)

- keep(c) = 1 − p(reject | c).
- role(c) = argmax [α · p_v(role | c)/Σ + (1 − α) · role_mass(c)/Σ].
- type(window) = argmax_t [log p(t) + λ · Σ_kept log P(role | t)], with P(role | t) from **train** gold, add-1 smoothed.
- Candidates are visited by decreasing keep; one is accepted if keep ≥ θ, length ≥ min_len, and it has no token overlap with an accepted span (collision rule `any`).

Grid searched on dev:
- α ∈ {0, .25, .5, .75, 1}
- λ ∈ {0, .25, .5, .75, 1, 1.5}
- collision ∈ {any, iou, wis}
- min_len ∈ {1, 2, 3}
- θ ∈ {.20, .30, .35, …, .70, .80}

**Frozen rule** (`assets/decoding_rule.json`, frozen 2026-09-23 19:31:30): θ 0.20, min_len 1, `any`, λ 0.5, α 0.25; dev Arg-C 49.16.

### 2.5 Baseline without verifier (`scripts/proposer_baseline.py`)

CARVE-simple decoded with its calibrated threshold: keep a role span if its mean posterior ≥ τ and its length ≥ min_len.
- The rule is frozen on dev as the τ/min_len maximising the **mean** dev Arg-C over seeds 42 / 13 / 101.
- Grid: τ ∈ {0.5 … 0.95}, min_len ∈ {2, 3, 4}.
- Frozen rule: τ 0.90, min_len 3 (the same as the original CARVE). Dev: 48.10 ± 0.96 (49.19 / 47.39 / 47.72).

## 3. Protocol and test-look log

Test-look log (all on 2026-09-23 unless stated):

1. The original CARVE was trained, frozen and evaluated on test (earlier work, 50.48 ± 1.15).
2. HONE was developed on dev with the original CARVE as proposer. Its multi-seed configuration was frozen at 16:41 and evaluated on test once (53.06 ± 1.04). A decoding ablation and a pooled-threshold control were then evaluated on test, with rules re-frozen on dev and no selection by test.
3. **CARVE-simple:**
   - its 3 seeds were trained, and its rule was frozen on dev (τ 0.90, min_len 3);
   - it was then evaluated on test once: 50.73 ± 0.19 Arg-C.
4. **HONE with the original CARVE proposer, single seed:** rule frozen on dev at 18:46:40, test 53.35.
5. **CLAVE, single seed** (this system):
   - 5 out-of-fold proposers were trained, then the verifier;
   - rule frozen on dev at **19:31:30**;
   - test predictions written at 19:31:40, i.e. after the freeze;
   - result: test 52.36.
6. **Selection of the final configuration.** CLAVE was selected as the release configuration after steps 2–5 had been observed on test. It is not the best of them by test score, so the choice did not inflate the reported number. It is nonetheless a post-hoc choice.

Throughout:
- every model, epoch and decoding parameter was chosen on dev;
- no test label was used for training, tuning or feature construction.

## 4. Results (test)

### 4.1 Trigger ROUGE-L

| Method | P | R | F1 |
|---|---|---|---|
| EEQA | 81.93 | 34.57 | 45.05 |
| DEGREE | 64.56 | 63.49 | 56.85 |
| OneIE | 73.73 | 79.40 | 72.40 |
| GPT 0-shot | 65.38 | 72.73 | 67.57 |
| GPT 1-shot | 72.67 | 77.77 | 74.05 |
| GPT 2-shot | 73.38 | 78.45 | 74.76 |
| GPT 5-shot | 73.70 | 78.82 | 75.08 |
| Qwen 2-shot | 57.27 | 69.71 | 61.18 |
| Llama 0-shot | 54.88 | 61.07 | 55.83 |
| DS-R1-Llama 1-shot | 41.81 | 41.94 | 40.72 |
| original CARVE (3 seeds) | 83.85 ± 1.25 | 76.28 ± 0.55 | 76.93 ± 0.80 |
| CARVE-simple (3 seeds) | 84.77 ± 0.43 | 76.15 ± 1.21 | 77.22 ± 0.81 |
| CARVE-simple, seed 42 | 85.13 | 76.78 | 77.82 |
| **CLAVE** | **85.13** | **76.78** | **77.82** |

ROUGE-L of CLAVE equals that of its proposer by construction.

### 4.2 Argument extraction, IoU > 0.5

| Method | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| EEQA | 32.09 | 33.77 | 32.91 | 25.85 | 27.20 | 26.51 |
| DEGREE | 67.79 | 19.13 | 29.84 | 48.99 | 13.83 | 21.57 |
| OneIE | 51.11 | 56.29 | 53.57 | 39.69 | 43.71 | 41.61 |
| GPT 0-shot | 43.03 | 55.56 | 48.50 | 30.40 | 39.25 | 34.26 |
| GPT 1-shot | 50.14 | 50.22 | 50.18 | 34.60 | 34.66 | 34.63 |
| GPT 2-shot | 49.12 | 51.29 | 50.18 | 33.99 | 35.49 | 34.72 |
| GPT 5-shot | 50.04 | 49.93 | 49.98 | 34.51 | 34.42 | 34.47 |
| Qwen 5-shot | 46.94 | 31.36 | 37.60 | 21.67 | 14.48 | 17.36 |
| Llama 1-shot | 44.70 | 34.08 | 38.68 | 18.93 | 14.44 | 16.38 |
| DS-R1-Llama 1-shot | 42.62 | 17.67 | 24.98 | 19.59 | 8.12 | 11.48 |
| original CARVE (3 seeds) | 63.46 ± 2.16 | 53.35 ± 2.98 | 57.95 ± 2.46 | 55.30 ± 1.13 | 46.47 ± 1.74 | 50.48 ± 1.15 |
| CARVE-simple (3 seeds) | 62.64 ± 0.23 | 55.03 ± 1.08 | 58.58 ± 0.57 | 54.24 ± 0.46 | 47.65 ± 0.65 | 50.73 ± 0.19 |
| CARVE-simple, seed 42 | 62.91 | 54.41 | 58.35 | 54.66 | 47.28 | 50.70 |
| **CLAVE** | **70.18** | 52.53 | **60.09** | **61.15** | 45.78 | **52.36** |
| Δ vs OneIE | +19.07 | −3.76 | +6.52 | +21.46 | +2.07 | +10.75 |
| Δ vs CARVE-simple, seed 42 | +7.27 | −1.88 | +1.74 | +6.49 | −1.50 | +1.66 |

Counts for CLAVE:
- Arg-C: 244 matched / 399 predicted / 533 gold.
- Arg-I: 280 matched.

### 4.3 All four matching modes

| mode | ArgI-P | ArgI-R | ArgI-F1 | ArgC-P | ArgC-R | ArgC-F1 |
|---|---|---|---|---|---|---|
| Exact Match | 46.37 | 34.71 | 39.70 | 42.61 | 31.89 | 36.48 |
| Simple overlap | 81.45 | 60.98 | 69.74 | 69.42 | 51.97 | 59.44 |
| SciREX > 0.5 | 76.19 | 57.04 | 65.24 | 65.41 | 48.97 | 56.01 |
| **IoU > 0.5** | 70.18 | 52.53 | 60.09 | 61.15 | 45.78 | 52.36 |

Same modes for CARVE-simple, 3-seed mean (ArgC-F1): EM 34.23 ± 1.27, overlap 59.85 ± 0.43, SciREX 56.26 ± 0.44, IoU 50.73 ± 0.19.

### 4.4 CARVE-simple per seed (rule τ 0.90, min_len 3)

| seed | RgL P / R / F1 | ArgI P / R / F1 | ArgC P / R / F1 |
|---|---|---|---|
| 42 | 85.13 / 76.78 / 77.82 | 62.91 / 54.41 / 58.35 | 54.66 / 47.28 / 50.70 |
| 13 | 84.29 / 76.92 / 77.55 | 62.50 / 56.29 / 59.23 | 53.75 / 48.41 / 50.94 |
| 101 | 84.88 / 74.76 / 76.30 | 62.50 / 54.41 / 58.17 | 54.31 / 47.28 / 50.55 |

### 4.5 Dev (frozen rules)

| system | dev Arg-C IoU F1 |
|---|---|
| original CARVE, 3 seeds | 47.14 ± 0.26 |
| CARVE-simple, 3 seeds | 48.10 ± 0.96 |
| CARVE-simple, seed 42 | 49.19 |
| **CLAVE** | **49.16** |
| CLAVE, verifier-only rule (α = 1, λ = 0, training-time tuning) | 46.69 |

- On dev the verifier does not add to its proposer (49.16 vs 49.19). The test gain (+1.66) is inside sampling noise.
- **Tie note:** HONE with the original CARVE proposer has exactly the same dev score (49.16). Both runs happen to match 219 of 394 predicted spans against 497 gold. This was checked: each run used its own candidate set (795 vs 792 candidates); it is not a file mix-up.

### 4.6 Paired bootstrap (`scripts/bootstrap_ci.py`, 5,000 resamples, same windows for both sides)

| system vs reference | metric | Δ | 95 % CI | P(Δ > 0) |
|---|---|---|---|---|
| CLAVE vs CARVE-simple s42 | Arg-C | +1.66 | [−0.14, +3.44] | 0.962 |
| | Arg-I | +1.74 | [−0.65, +4.03] | 0.920 |
| CLAVE vs original CARVE s42 | Arg-C | +3.14 | [−0.62, +6.83] | 0.951 |
| | Arg-I | +4.84 | [+0.86, +8.83] | 0.992 |
| CLAVE vs HONE on original-CARVE proposer (1 seed) | Arg-C | −0.99 | [−4.41, +2.17] | 0.287 |
| | Arg-I | −0.06 | [−3.72, +3.30] | 0.483 |
| CARVE-simple (3 seeds) vs original CARVE (3 seeds) | Arg-C | +0.25 | [−1.66, +2.12] | 0.603 |
| | Arg-I | +0.64 | [−1.27, +2.46] | 0.746 |

Absolute 95 % CIs for CLAVE: Arg-C [47.28, 57.18], Arg-I [55.12, 64.72]. Both lower bounds exceed OneIE (41.61 and 53.57).

Test event-type accuracy: 88.96 % for both CLAVE and CARVE-simple seed 42. The joint type decision did not change any test window's type.

## 5. Development evidence behind each design decision

Each row is labelled with the configuration it was measured on. Rows marked *orig.* used the original two-head CARVE proposer, seed 42. They were **not** re-run on CARVE-simple.

| decision | evidence (dev Arg-C unless stated) | configuration |
|---|---|---|
| calibrated threshold instead of argmax | 38.88 → 48.00 (+9.12) | original CARVE |
| single merged head instead of two heads | +0.75 ± 1.6 (inert) | original CARVE, 3 seeds |
| no type conditioning | −0.24 ± 1.6 (inert) | original CARVE, 3 seeds |
| both removed (= CARVE-simple) | 48.50 ± 0.68 vs 47.58 ± 0.55 (same grid); test Δ +0.25 [−1.66, +2.12] | 3 seeds |
| no CRF / span-level role head / LLRD | −0.28 / −0.96 / −2.20 | original CARVE, 3 seeds |
| out-of-fold instead of in-sample verifier training | 49.16 vs 44.87 (−4.3); in-sample candidates 97.5 % positive | *orig.* |
| proposer features in the verifier | 49.16 vs 48.24 text-only (+0.9) | *orig.* |
| hybrid role (α) | verifier keep AUC 0.890 vs proposer 0.860, but role accuracy 77.1 % vs 81.3 % | *orig.* |
| no candidate composition | 48.82 vs 49.16 (rejected) | *orig.* |
| no verifier ensembling | +0.10 (rejected) | *orig.*, pooled seeds |

## 6. Other configurations (archived, for comparison only)

| configuration | test Arg-C | test Arg-I | note |
|---|---|---|---|
| original CARVE, 3 seeds | 50.48 ± 1.15 | 57.95 | previous release |
| CARVE-simple, 3 seeds | 50.73 ± 0.19 | 58.58 | this proposer, no verifier |
| HONE on original-CARVE proposer, 1 seed | 53.35 | 60.15 | +4.13 [+1.57, +6.74] over its own proposer |
| HONE on 3 pooled original-CARVE proposers, 3 verifiers | 53.06 ± 1.04 | 63.05 ± 0.27 | best Arg-I; a pooled-threshold control without verifier scored 53.87 |
| **CLAVE, 1 seed** | **52.36** | **60.09** | released |

## 7. Audit

| check | result |
|---|---|
| window overlap train∩dev, train∩test, dev∩test | 0 / 0 / 0 |
| abstracts of test windows also present in train | 143 / 147 (97.3 %). The official split is window-level; shared by all published baselines. |
| `sent_id` as input | never. Its suffix predicts the gold type 94.9 / 98.7 / 93.9 % (train / dev / test); it is annotation metadata. |
| gold event type at inference | never (predicted; 88.96 % test accuracy) |
| test predictions: windows | 163 / 163 |
| predicted vs gold arguments | 399 / 533: under-prediction, so no precision gaming by volume |
| duplicate (span, role) pairs in predictions | 0 |
| rule timestamp vs first test prediction | 19:31:30 < 19:31:40 |
| benchmark files modified | 0 |

## 8. Reproducibility checks performed on the released code

- **Contract tests:** pass. The oracle scores 100.00 on dev and test.
- **Proposer inference:** from the reported checkpoint, dev and test candidate files are byte-identical to the ones the reported verifier used.
- **Baseline test predictions:** byte-identical for seeds 42, 13 and 101. The metrics match §4.4.
- **Verifier test predictions:** byte-identical to the reported ones. Re-running `freeze` reproduces the frozen rule.
- **Training determinism:** GPU training is nondeterministic. Two runs of the development code differ after one epoch (loss 2.5878 vs 2.5891), and the released code lands in the same range (2.589). Retraining therefore reproduces the result up to seed-level variation, not bit for bit.
- **Run names:** the reported verifier run was called `hs_v42_ps42` in development; it is `runs/verifier_s42` here. The field `runs` inside `assets/decoding_rule.json` keeps the original name.

## 9. Provenance

Development records live in the SciEvent research repository:
- `method/HONE/`: `RESEARCH_LOG.md` stages 0–13, `HYPOTHESES.md`, `DECISIONS.md`, `CARVE_SIMPLE_REPORT.md`, logs, predictions;
- `method/CARVE-full/`: the previous release of this repository with its ablations.

## 10. Registered three-seed extension (2026-09-24)

### 10.1 Protocol and provenance

The three independent pairs are proposer/verifier seeds **42/42, 13/13, and
101/101**. A single decoding rule was selected by **mean dev Arg-C IoU F1**
across the three verifier runs and frozen at 2026-09-24 13:45:39 before the
registered T-main test look. The rule in
`experiment/assets/decoding_rule_3seed.json` is θ 0.30, min_len 3, NMS `any`,
role α 0.50, type λ 0.50. The original one-seed release still uses
`assets/decoding_rule.json` (θ 0.20, min_len 1, α 0.25, λ 0.50).

The complete pre-registration is `experiment/EXPERIMENT_PLAN.md`. Every new
test look, rule timestamp, output path and result is in
`experiment/TEST_LOG.md`. T-main, T-A1, T-A2, T-A3–A7a/b and T-P1 were each
evaluated once under their own frozen rules. Optional T-P2a/b was explicitly
authorized by the user for Phase 7. The Phase-2 candidate diagnostic that
looked at test gold labels outside the plan is retained in TEST_LOG as a
**post-hoc mistake**; it did not select a checkpoint, rule, or model variant.
The original release was itself chosen after earlier test comparisons, as
described in §3; this extension does not erase that history.

### 10.2 Main results

All figures below are **mean ± sample standard deviation** across the three
pairs. The complete P/R/F1 tables are generated from `experiment/results/*.json`
by `experiment/scripts/tables.py`: `experiment/tables/T1.md` (trigger),
`T2.md` (arguments), `T3.md` (four matching modes), and `T4.md` (bootstrap).

| System | Test Arg-C IoU F1 | Test Arg-I IoU F1 | Trigger ROUGE-L F1 | Rule |
|---|---:|---:|---:|---|
| Original CLAVE, one seed | 52.36 | 60.09 | 77.82 | `assets/decoding_rule.json` |
| **CLAVE, three seeds** | **52.45 ± 0.73** | **60.49 ± 1.25** | **77.22 ± 0.81** | `experiment/assets/decoding_rule_3seed.json` |
| CARVE-simple, three seeds | 50.73 ± 0.19 | 58.58 ± 0.57 | 77.22 ± 0.81 | `assets/proposer_rule.json` |
| Original CARVE, three seeds (archive) | 50.48 ± 1.15 | 57.95 ± 2.46 | 76.93 ± 0.80 | archived rule |

CLAVE's shared-rule dev Arg-C is **48.43 ± 0.47**, versus CARVE-simple
**48.10 ± 0.96**. The registered paired-window bootstrap (5,000 resamples of
the 163 test windows, averaging three runs on each side) gives CLAVE minus
CARVE-simple **+1.72 [+0.24, +3.30]** Arg-C and **+1.90 [+0.21, +3.72]**
Arg-I. Against original CARVE, Arg-C is +1.97 [−0.20, +4.07] and Arg-I is
+2.54 [+0.45, +4.65]. The absolute CLAVE confidence intervals are
[47.87, 57.08] Arg-C and [56.22, 64.61] Arg-I. OneIE supplies only published
aggregate scores here (41.61 Arg-C, 53.57 Arg-I), so no paired comparison
against its per-window predictions is available. CLAVE's trigger score equals
its own proposer's score by construction.

### 10.3 Registered hypotheses and ablations

The pre-registered support threshold is **CLAVE minus comparator ≥ +0.5 dev
Arg-C**. Test comparisons are reported separately and do not overturn the
dev decision. The full ablation table with P/R/F1, rules and test bootstrap
intervals is `experiment/tables/T5.md`.

| Hypothesis | Dev contrast | Decision | Test contrast (CLAVE minus comparator), 95% CI |
|---|---:|---|---:|
| G1, verifier vs CARVE-simple | +0.33 | rejected | +1.72 [+0.24, +3.30] |
| G2, cross-fitting vs in-sample | +1.63 | supported | +3.48 [+2.10, +4.92] |
| G3, proposer features vs text-only | −0.66 | rejected | +0.99 [0.00, +2.03] at reported precision |
| G4, role mixing vs α=1 | +1.07 | supported | +2.46 [+0.56, +4.35] |
| G5, type combination vs λ=0 | +0.08 | rejected | +0.15 [0.00, +0.48] at reported precision |

The text-only and λ=0 test intervals touch zero at the reported precision.
NMS `iou` and `wis` have the same scores as `any` in all three runs. `iou`
predictions are byte-identical to `any`; `wis` changes record ordering/IDs but
not the per-window event-type/role/span sets. This is a descriptive identity,
not evidence for a gain from a different NMS rule.

### 10.4 Phase-7 proposer and error analysis

The fixed argmax proposer rule (τ 0, min_len 1; T-P1) scores **38.32 ± 0.25**
dev and **39.67 ± 0.76** test Arg-C. The calibrated CARVE-simple rule scores
**48.10 ± 0.96** dev and **50.73 ± 0.19** test. The full proposer-design
table, including the historical same-grid ablations, is
`experiment/tables/T6.md`.

The descriptive test analysis in `experiment/tables/T7.md` reports role,
event-type and domain P/R/F1, gold-span-length recall, error buckets, and
precision/recall shifts. For example, Arg-C precision rises **+6.52 points**
and recall falls **0.94 points** versus CARVE-simple, averaged over three
paired runs. The dev oracle keep+role values under the *new* shared rule are
69.91 / 73.47 / 72.44 for seeds 42/13/101; the earlier seed-42 value 72.20
in §2 used the old one-seed rule and is not directly comparable. T7 separates
the event-type-only oracle at decoding from the joint keep+role+type oracle.
`experiment/tables/T8.md` records candidate counts, positive rates, parameter
counts and elapsed training times. Test descriptive analysis was generated
before the optional T-P2 evaluation, a Phase-7 ordering deviation; no T7
value was used to choose T-P2 training, rules or checkpoints.

### 10.5 Proposer design, 2×2 factorial on test (T-P2a/b)

The two middle cells were retrained with the archived CARVE code in an isolated
copy (`experiment/archive_carve_full/`). The training code is byte-identical to
the archive (`experiment/results/p2_code_provenance.json`). Each row has one
rule frozen on mean dev Arg-C over seeds 42/13/101 before its registered test
look.

| design | dev Arg-C | test Arg-I F1 | test Arg-C F1 | Arg-C Δ vs CARVE-simple [95 % CI] |
|---|---:|---:|---:|---:|
| two heads + type conditioning (original CARVE, archive) | 47.14 ± 0.26 | 57.95 ± 2.46 | 50.48 ± 1.15 | −0.25 [−2.12, +1.66] |
| two heads, no conditioning (T-P2a; τ 0.90, ml 3) | 47.13 ± 1.32 | 57.28 ± 0.73 | 49.82 ± 0.62 | −0.91 [−2.73, +0.97] |
| one head + conditioning (T-P2b; τ 0.88, ml 3) | 48.59 ± 3.13 | 58.48 ± 0.56 | 50.76 ± 0.39 | +0.03 [−1.41, +1.51] |
| **one head, no conditioning (CARVE-simple)** | 48.10 ± 0.96 | 58.58 ± 0.57 | 50.73 ± 0.19 | — |

- No cell differs from CARVE-simple on test; every interval covers zero. This confirms on test what the dev ablations showed: both removed components are inert. CARVE-simple is the simplest cell and has the lowest test variance.
- Calibration is the proposer component that matters: argmax decoding (T-P1) is −11.07 Arg-C [−13.36, −8.85] against the calibrated rule.
- Bootstrap values are in `experiment/results/bootstrap_phase7.json`.
- **Execution note (logged in TEST_LOG):** the first T-P2a test pass crashed after scoring seed 42. The archive decoder writes `metrics.test.json` into a `preds/` directory it does not create, and no prediction file had been written. The same frozen rule and checkpoints were re-run once to completion, so it remains the same registered look. `experiment/scripts/reproduce_p2.sh` now creates the directory first.
- After recording the results, the six T-P2 checkpoints were deleted (plan §8). Their logs, dev predictions and test predictions are kept.

### 10.6 Seed-level operating points

Under the shared rule (θ 0.30), the three runs reach similar Arg-C F1 at different precision/recall operating points:

| pair | Arg-C P / R / F1 |
|---|---|
| 42/42 | 64.36 / 45.40 / 53.25 |
| 13/13 | 52.40 / 51.22 / 51.80 |
| 101/101 | 65.54 / 43.53 / 52.31 |

- The seed-13 verifier was selected at epoch 1 on dev (48.02), before its keep scores had sharpened. It therefore keeps more spans: 890 predicted arguments, against 751 and 733.
- This is why the three-seed P and R standard deviations (±7.27 / ±4.01 on Arg-C) are much larger than the F1 standard deviation (±0.73). Report it with any P/R table.
