"""Proposer inference and its threshold decoder.

`posteriors` runs a trained CARVE-simple checkpoint and returns per-word
posteriors over the merged 13-type BIO inventory plus window event-type
posteriors. Two consumers:

  * `scripts/propose.py` turns them into candidate spans for the verifier;
  * `scripts/proposer_baseline.py` decodes them directly with the calibrated
    threshold rule below (the CARVE-simple baseline, no verifier).

Threshold rule: keep a scored-role span iff its mean token posterior >= tau and
its length >= min_len. Both are tuned on dev only and frozen before test.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from clave.data import AAO_TYPES, EVENT_TYPES, MERGED_LABEL2ID, ROLE_TYPES, bio_to_spans
from clave.evaluate import to_oneie_record
from clave.model import CarveTagger
from clave.train import WindowDataset, collate

ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}


@torch.no_grad()
def posteriors(ckpt, model_name, windows, tok, max_len=640, batch_size=16, device="cuda"):
    """-> (span_post: list of [n_words, 27] arrays, type_post: [n_windows, 4])."""
    dl = DataLoader(WindowDataset(windows, tok, max_len), batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    span_p = [None] * len(windows)
    type_p = np.zeros((len(windows), len(EVENT_TYPES)), dtype=np.float64)
    model = CarveTagger(model_name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for batch in dl:
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp):
            tl, sl = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"])
        tsm = torch.softmax(tl.float(), -1).cpu().numpy()
        ssm = torch.softmax(sl.float(), -1).cpu().numpy()
        for j, idx in enumerate(batch["idx"].tolist()):
            nv = int(batch["word_mask"][j].sum())
            type_p[idx] = tsm[j]
            span_p[idx] = ssm[j, :nv]
    return span_p, type_p


def spans_with_conf(p, id2label=ID2MERGED):
    """argmax decode + mean-posterior confidence per span: (s, e, type, conf)."""
    ids = p.argmax(-1).tolist()
    return [(s, e, t, float(np.mean([p[i, ids[i]] for i in range(s, e)]))) for s, e, t in bio_to_spans(ids, id2label)]


def apply_decoding_rules(spans, tau, min_len):
    """Keep (s, e, type, conf) spans with conf >= tau and length >= min_len."""
    return [(s, e, t) for s, e, t, c in spans if c >= tau and e - s >= min_len]


def threshold_records(windows, span_p, type_p, tau, min_len):
    """CARVE-simple baseline predictions: thresholded role spans, AAO/trigger
    spans from the same tagger, window type = argmax of the type head."""
    recs = []
    for i, w in enumerate(windows):
        n = len(w.tokens)
        spans = spans_with_conf(span_p[i])
        roles = apply_decoding_rules([x for x in spans if x[2] in ROLE_TYPES], tau, min_len)
        aao = [x for x in spans if x[2] in AAO_TYPES]
        trig = next(((s, e) for s, e, t, _ in aao if t == "Action"), (0, min(1, n)))
        recs.append(to_oneie_record(w.sent_id, w.tokens, EVENT_TYPES[int(type_p[i].argmax())], roles,
                                    [(s, e, t) for s, e, t, _ in aao if t != "Action"], trig))
    return recs
