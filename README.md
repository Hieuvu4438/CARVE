# CARVE

**C**lause-level **A**rgument **R**ecovery **V**ia s**E**gmentation — a span-segmentation
model for scientific event argument extraction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of *"CARVE: Clause-Level Argument Recovery via Segmentation
for Multi-Domain Scientific Event Extraction"*, evaluated on the
[SciEvent](https://aclanthology.org/2025.emnlp-main.871/) benchmark (Dong et al., EMNLP 2025).

---

## TL;DR

Every tuned baseline on SciEvent (OneIE, DEGREE, EEQA) models event arguments the way
ACE-style event extraction does: short entity mentions found by graph decoding, template
generation, or span-QA. **We measured the benchmark before choosing an architecture, and
the data says otherwise.** Each annotated window contains exactly one event; the scored
arguments are **10–15-token clauses**; **97.6 %** of windows have no overlap at all between
scored arguments; and the most common gap between two consecutive arguments is **zero
tokens**.

SciEvent argument extraction is therefore **clause-level span segmentation with semantic-role
labelling**, not mention extraction. CARVE models it that way and adds a three-parameter
metric-aware decoding rule tuned on dev only.

---

## Results

Frozen evaluation on the official test split, three seeds, scored with the benchmark's own
evaluator. Baseline numbers are as reported in the SciEvent paper.

| Method | Arg-I IoU F1 | **Arg-C IoU F1** | Trigger ROUGE-L F1 |
|---|---|---|---|
| EEQA | 32.91 | 26.51 | 45.05 |
| DEGREE | 29.84 | 21.57 | 56.85 |
| OneIE *(previous best)* | **53.57** | **41.61** | 72.40 |
| GPT-4o (5-shot) | 49.98 | 34.47 | **75.08** |
| **CARVE (ours)** | **57.95 ± 2.46** | **50.48 ± 1.15** | **76.93 ± 0.80** |
| | **+4.38** | **+8.87** | **+1.85** |

Every individual seed beats every target. Full per-domain, per-role, per-event-type and
all-matching-mode tables are in [`docs/PAPER_VI.md`](docs/PAPER_VI.md) and
[`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md).

### An honest decomposition

Neither half of the method beats OneIE on its own — we state this explicitly rather than
letting the headline imply otherwise:

| Stage | Arg-C P | Arg-C R | Arg-C F1 |
|---|---|---|---|
| OneIE | 39.69 | 43.71 | 41.61 |
| CARVE, **uncalibrated** argmax | 32.43 | **51.83** | 39.89 |
| CARVE, **+ calibrated decoding** | 55.30 | 46.47 | **50.48** |

What the reframing buys is **recall** (+8.12 over OneIE). Calibration converts that
headroom into F1; it cannot create recall, only trade it. Both parts are load-bearing.

---

## Method

```
Window text (whitespace tokens)
        │
        ▼
  DeBERTa-v3-large ──► first-sub-token pooling to word states
        │
        ├──► TYPE head : attention-pooled → 4-way window event type
        │                └──► learned type embedding, added back to word states
        ├──► ROLE head : word-level BIO over the 9 scored semantic roles
        │                (on the type-conditioned states)
        └──► AAO  head : word-level BIO over ⟨Agent, Action, Primary/SecondaryObject⟩
                         (on the unconditioned states; used only for ROUGE-L)
```

Four design decisions, each forced by a measurement:

| Decision | Evidence |
|---|---|
| BIO tagging rather than span enumeration | Measured ceiling of the formulation is **99.70** Arg-C IoU — it costs ~0.3 F1 |
| **Two** disjoint BIO heads, not one | Cross-group span overlap is **5.9 %**, within-group only ≤2.4 % |
| Explicit event-type head + conditioning | The metric is event-type sensitive with one event per window, so a wrong type annihilates every argument in it; oracle typing is worth **+4.72** Arg-C |
| Confidence-thresholded decoding | The raw tagger emits **784 spans against 497 gold** — a calibration failure, not a representation failure |

Frozen decoding rule: `tau = 0.90`, `min_len = 3`, `merge_gap = 0`
(see [`assets/decoding_rules.json`](assets/decoding_rules.json)).

---

## Installation

```bash
git clone https://github.com/Hieuvu4438/CARVE.git
cd CARVE
pip install -e .
```

Requires Python ≥ 3.9 and a CUDA GPU. All experiments in the paper were run on a single
NVIDIA RTX 5880 Ada (49 GB); a full training run takes **~9 minutes** and the entire paper
costs **~2.5 GPU-hours**.

> **Note on `transformers >= 5`.** It honours the dtype stored in a checkpoint, and
> `microsoft/deberta-v3-large` ships **fp16** weights, which makes AdamW produce NaN on the
> first optimizer step. CARVE loads the encoder with `dtype=torch.float32` and takes mixed
> precision from autocast only. If you reimplement this, you will hit the same trap.

## Data

CARVE treats the SciEvent benchmark as a **read-only** dependency and never modifies it.

```bash
bash scripts/setup_data.sh
```

This clones the benchmark into `third_party/SciEvent` at the pinned commit and runs the
benchmark's *own* preprocessing to produce the ONEIE-format splits that every baseline in
the paper is also trained and scored on (1278 / 158 / 163 windows).

If you already have a checkout, point at it instead:

```bash
export SCIEVENT_ROOT=/path/to/SciEvent
```

Verify the setup — this also checks that our prediction exporter is exact, by routing
**gold** spans through it into the official scorer:

```bash
python3 tests/test_contract.py
```

```
[ok] splits load with expected sizes
[ok] BIO round-trip recovers 3857/3897 = 0.9897 of gold role spans
[oracle:dev] ArgC-IoU 100.00  ArgI-IoU 100.00  ArgC-EM 100.00  RougeL 100.00
[oracle:test] ArgC-IoU 100.00  ArgI-IoU 100.00  ArgC-EM 100.00  RougeL 100.00
[bio-ceiling:dev] ArgC-IoU 99.70  ArgI-IoU 99.70  ArgC-EM 99.70  RougeL 100.00
```

> **CSCW abstracts** are not redistributable and are absent from the public benchmark
> release. To reproduce the exact numbers you must obtain them yourself; see the upstream
> README.

## Quick start

```bash
# Train one model (train -> dev, checkpoint selected on dev Arg-C IoU F1)
python3 -m carve.train -c configs/carve.json --set run_name=carve_s42 seed=42

# Evaluate with the frozen decoding rule
python3 -m carve.decode --ckpt runs/carve_s42/best.pt --split test \
    --rules assets/decoding_rules.json --out preds/test_s42.jsonl

# Error analysis
python3 -m carve.analysis --pred preds/test_s42.jsonl --split test
```

Any config key can be overridden inline:

```bash
python3 -m carve.train -c configs/carve.json --set seed=13 epochs=40 lr_encoder=2e-5
```

## Reproducing the paper

```bash
bash scripts/reproduce.sh
```

Runs the full protocol end to end (~35 minutes):

1. **Correctness tests** — data contract and official-scorer oracle.
2. **Training** — three seeds (42 / 13 / 101) on `train`, checkpoints selected on `dev`.
3. **Freezing the decoding rule** — `tau` and `min_len` chosen to maximise the **mean dev
   Arg-C IoU across seeds**, so the rule is never fitted to whichever seed happens to look
   best on dev.
4. **Test, evaluated exactly once** per seed with that single rule. Nothing changes after.
5. **Result tables and error analysis.**

### Ablations

All four reported ablations were rejected. The configs are provided so the negative results
are reproducible:

```bash
python3 -m carve.train -c configs/ablation_crf.json        # linear-chain CRF
python3 -m carve.train -c configs/ablation_llrd.json       # layer-wise LR decay
python3 -m carve.train -c configs/ablation_span_role.json  # span-level role re-classifier
```

Multi-seed posterior ensembling (also rejected — it helps argmax decoding but *hurts*
calibrated decoding, because averaging compresses the posterior distribution the threshold
relies on):

```bash
python3 -m carve.decode --ckpt runs/carve_s42/best.pt runs/carve_s13/best.pt \
    runs/carve_s101/best.pt --split dev --tune
```

### Leakage and protocol audit

```bash
bash scripts/audit.sh
```

Re-verifies our numbers through the benchmark's scorer run as a CLI, checks split overlap,
confirms the frozen rule predates every test prediction, and confirms we *under*-predict
(446 spans against 533 gold) rather than flooding the output to win precision.

The audit also documents a benchmark finding: the released `split_data.py` splits on
**`wnd_id`** (window), not `doc_id`, despite the paper describing an 80/10/10 split by
document — so 97.3 % of test documents have sibling segments in train. This applies
identically to every baseline, so the comparison stays like-for-like, but we measure its
effect directly with a document-disjoint re-split:

```bash
python3 scripts/make_docsplit.py                       # 394/45/61 docs, zero overlap
python3 -m carve.train -c configs/docsplit.json --set run_name=docsplit_s42 seed=42
```

Under that stricter split CARVE still reaches **44.93** Arg-C IoU, above OneIE's 41.61.

### Additional analyses

```bash
python3 scripts/type_headroom.py runs/carve_s42/best.pt   # oracle event-type headroom
python3 scripts/results_tables.py                         # all comparison tables
```

## Repository structure

```
carve/
├── paths.py        resolution of the read-only SciEvent benchmark paths
├── data.py         data contract, label inventories, BIO encode/decode
├── model.py        encoder + ROLE / AAO / TYPE heads
├── crf.py          optional linear-chain CRF (ablation; reported as negative)
├── train.py        training loop, checkpoint selection, prediction export
├── decode.py       posterior caching, calibrated decoding, frozen evaluation
├── evaluate.py     thin wrapper over the benchmark's OWN scorer
└── analysis.py     error taxonomy + per-role / domain / type / length tables

configs/            one JSON per experiment; carve.json is the frozen recipe
scripts/            data setup, reproduction, rule freezing, audit, tables
tests/              data contract and evaluator oracle tests
assets/             the frozen decoding rules
docs/               the paper and the full experiment record
```

## Metric integrity

We never reimplement a metric. `carve/evaluate.py` loads the benchmark's own
`EM_overlap_eval.py` and calls **its functions**, so matching semantics are bit-identical
to the published numbers: event-type-sensitive, trigger-insensitive,
`{Agent, PrimaryObject, SecondaryObject}` excluded from Arg-I/Arg-C, IoU > 0.5, and ROUGE-L
over the ⟨Agent, trigger, PrimaryObject, SecondaryObject⟩ tuple.

Running the upstream scorer as a CLI on our frozen test predictions reproduces our reported
numbers to the last decimal.

## Limitations

- **Short spans.** Recall at IoU > 0.5 is **14.3 %** for 1–4-token gold spans, against ~60 %
  for 10–34. This is self-inflicted by `min_len = 3` and `tau = 0.90`; a length-conditioned
  threshold should recover much of it.
- **Digital Humanities** scores 29.75 against 74.39 for computational biology. We reproduce
  the humanities gap the SciEvent paper reports and do **not** close it.
- **Tail roles score zero.** Analysis (10 test instances), Contradictions (1), Ethical (1).
- **Operating point favours precision.** Arg-I recall is 2.94 *below* OneIE's while
  precision is 12.35 above. OneIE is reported at its natural operating point and we at a
  dev-tuned one.
- **Segmentation is not modelled.** We consume the provided windows, exactly as OneIE,
  DEGREE and EEQA do, and report no segmentation number rather than an incomparable one.

## Documentation

| File | Contents |
|---|---|
| [`docs/PAPER_VI.md`](docs/PAPER_VI.md) | Full paper (Vietnamese): motivation, method, experiments, audit, limitations |
| [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md) | Complete experiment record — every measured number with the exact configuration that produced it |

## Citation

```bibtex
@misc{carve2026,
  title  = {CARVE: Clause-Level Argument Recovery via Segmentation
            for Multi-Domain Scientific Event Extraction},
  author = {Hieu Vu},
  year   = {2026},
  url    = {https://github.com/Hieuvu4438/CARVE}
}
```

Please also cite the benchmark:

```bibtex
@inproceedings{dong2025scievent,
  title     = {SciEvent: Benchmarking Multi-domain Scientific Event Extraction},
  author    = {Dong, Bofu and Shah, Pritesh and Sonawane, Sumedh and
               Banerjee, Tiyasha and Brady, Erin and Du, Xinya and Jiang, Ming},
  booktitle = {Proceedings of EMNLP},
  year      = {2025},
  url       = {https://aclanthology.org/2025.emnlp-main.871/}
}
```

## Acknowledgements

This work builds on the [SciEvent benchmark](https://github.com/desdai/SciEvent) by Dong et
al. We use their data, their preprocessing scripts and their evaluator unmodified. The
encoder is [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) (He et al.,
MIT licence).

## License

[MIT](LICENSE).
