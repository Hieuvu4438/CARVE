"""Correctness tests: data contract, BIO round-trip, evaluator oracle, and the
candidate / decoding logic. CPU only.

The critical test is `test_oracle_upper_bound`: gold spans routed through our own
record-building code into the *official* upstream evaluator must score ~100. If
not, the prediction export is broken and every downstream number is meaningless.
"""

import os

import numpy as np

from clave import candidates as C
from clave.data import MERGED_LABEL2ID, ROLE_TYPES, bio_to_spans, default_data_dir, load_split, spans_to_bio
from clave.evaluate import score, to_oneie_record
from clave.select import decode_window

ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}


def test_splits_load():
    for sp, n in [("train", 1278), ("dev", 158), ("test", 163)]:
        assert len(load_split(sp)) == n, sp
    print("[ok] splits load with expected sizes")


def test_merged_bio_roundtrip():
    """The proposer's single 13-type inventory must recover almost every gold role span."""
    recovered = total = 0
    for x in load_split("train"):
        y = spans_to_bio(list(x.role_spans) + list(x.aao_spans), len(x.tokens), MERGED_LABEL2ID)
        got = set(bio_to_spans(y, ID2MERGED))
        total += len(x.role_spans)
        recovered += sum(s in got for s in x.role_spans)
    rate = recovered / total
    print(f"[ok] merged BIO round-trip recovers {recovered}/{total} = {rate:.4f} of gold role spans")
    assert rate > 0.95, rate


def test_oracle_upper_bound():
    """Gold spans routed through our exporter + the official evaluator = ceiling."""
    for split in ["dev", "test"]:
        recs = []
        for x in load_split(split):
            trig = next(((s, e) for s, e, t in x.aao_spans if t == "Action"), (0, 1))
            aao = [(s, e, t) for s, e, t in x.aao_spans if t != "Action"]
            recs.append(to_oneie_record(x.sent_id, x.tokens, x.event_type, x.role_spans, aao, trig))
        m = score(recs, os.path.join(default_data_dir(), f"{split}.oneie.json"))
        print(f"[oracle:{split}] ArgC-IoU {m['arg_c_iou']['f1']:.2f}  ArgI-IoU {m['arg_i_iou']['f1']:.2f}  "
              f"RougeL {m['trigger_rougeL']['f1']:.2f}")
        assert m["arg_c_iou"]["f1"] > 99.0 and m["arg_c_exact"]["f1"] > 99.0


def test_candidate_extraction_and_labels():
    """Argmax spans of a synthetic merged posterior; only scored roles become
    candidates; labels follow IoU > 0.5 against gold."""
    labs = ["B-Method", "I-Method", "I-Method", "O", "B-Agent", "B-Results", "I-Results"]
    p = np.full((len(labs), len(ID2MERGED)), 0.01)
    for i, l in enumerate(labs):
        p[i, MERGED_LABEL2ID[l]] = 0.9
    cands = C.extract(p, ID2MERGED, seed=0)
    assert [(c["s"], c["e"], c["role"]) for c in cands] == [(0, 3, "Method"), (5, 7, "Results")]
    assert len(cands[0]["role_mass"]) == len(ROLE_TYPES)

    class W:  # minimal window
        role_spans = [(0, 4, "Method")]
    C.label(W, cands)
    assert cands[0]["label"] == "Method" and cands[1]["label"] == C.REJECT
    print("[ok] candidate extraction and labelling")


def test_decoder_collisions():
    """Greedy decoding keeps the higher-scoring of two overlapping candidates."""
    cands = [{"s": 0, "e": 6}, {"s": 2, "e": 5}, {"s": 8, "e": 12}]
    probs = np.zeros((3, len(C.LABELS)))
    probs[0, 0], probs[0, C.LABEL2ID["Method"]] = 0.1, 0.9
    probs[1, 0], probs[1, C.LABEL2ID["Context"]] = 0.3, 0.7
    probs[2, 0], probs[2, C.LABEL2ID["Results"]] = 0.9, 0.1
    out = decode_window(cands, probs, theta=0.5, min_len=1, nms="any")
    assert out == [(0, 6, "Method")], out
    print("[ok] collision-aware decoding")


if __name__ == "__main__":
    test_splits_load()
    test_merged_bio_roundtrip()
    test_oracle_upper_bound()
    test_candidate_extraction_and_labels()
    test_decoder_collisions()
    print("\nALL CONTRACT TESTS PASSED")
