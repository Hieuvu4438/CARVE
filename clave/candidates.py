"""Candidate argument spans: extraction from proposer posteriors, normalisation
into verifier rows, and labelling against gold.

A *candidate* is one scored-role span the proposer (CARVE-simple) decodes at
argmax, with no confidence threshold. HONE replaces the proposer's hand-set
threshold with a learned decision, so candidates are as permissive as the
argmax decode allows and carry the proposer's evidence as features:

    s, e          span in whitespace-token space, end exclusive
    role          role the proposer assigned (argmax BIO type)
    conf          mean posterior of the assigned label over the span (the threshold score)
    role_mass     mean over the span of P(B-r) + P(I-r) for each of the 9 roles
    o_mass        mean over the span of P(O)
    seeds         proposer seed(s) that produced exactly this (s, e)

`pool_rows` normalises per-window rows (adds conf_max and n_agree). It accepts
several proposer seeds and merges identical spans; the released configuration
uses a single proposer seed, for which pooling only adds those two fields.

Labels follow the metric exactly: a candidate's target is the role of the gold
span it overlaps with IoU > 0.5, else "reject"; if two gold spans exceed 0.5
(overlapping gold, rare), the higher IoU wins.
"""

import json
import os

import numpy as np

from clave.data import ROLE_TYPES

REJECT = "reject"
LABELS = [REJECT] + list(ROLE_TYPES)          # verifier output space (10 classes)
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / (max(a[1], b[1]) - min(a[0], b[0]))


def extract(role_post, id2label, seed):
    """All argmax BIO spans of one window with their evidence features.

    role_post: [n_words, 1 + 2*len(ROLE_TYPES)] posterior over role BIO labels.
    Also accepts the single-head proposer's merged 13-type posterior (pass its
    id2label): spans are decoded over the full label space, exactly as that
    proposer decodes itself, and only spans of the nine scored roles are kept.
    """
    ids = role_post.argmax(-1).tolist()
    col = {l: i for i, l in id2label.items()}
    out = []
    cur_t, cur_s = None, None

    def close(end):
        if cur_t is None or cur_t not in ROLE_TYPES:
            return
        seg = role_post[cur_s:end]
        conf = float(np.mean([role_post[i, ids[i]] for i in range(cur_s, end)]))
        mass = [float(seg[:, col["B-" + r]].mean() + seg[:, col["I-" + r]].mean()) for r in ROLE_TYPES]
        out.append({"s": cur_s, "e": end, "role": cur_t, "conf": conf,
                    "role_mass": mass, "o_mass": float(seg[:, col["O"]].mean()), "seeds": [seed]})

    for i, lid in enumerate(ids):
        lab = id2label[lid]
        if lab == "O":
            close(i)
            cur_t, cur_s = None, None
            continue
        prefix, t = lab.split("-", 1)
        if prefix == "B" or cur_t != t:
            close(i)
            cur_t, cur_s = t, i
    close(len(ids))
    return out


def pool(per_seed):
    """Union of several seeds' candidates for one window.

    Exactly identical (s, e) spans are merged: their evidence is averaged and
    `seeds` records who proposed them, which gives the verifier an agreement
    signal. Near-duplicates (overlapping but not identical) stay separate.
    """
    merged = {}
    for cands in per_seed:
        for c in cands:
            key = (c["s"], c["e"])
            if key not in merged:
                merged[key] = {**c, "seeds": list(c["seeds"]),
                               "_conf": [c["conf"]], "_mass": [c["role_mass"]], "_o": [c["o_mass"]],
                               "_roles": [c["role"]]}
            else:
                m = merged[key]
                m["seeds"] += c["seeds"]
                m["_conf"].append(c["conf"]); m["_mass"].append(c["role_mass"])
                m["_o"].append(c["o_mass"]); m["_roles"].append(c["role"])
    out = []
    for m in merged.values():
        mass = np.mean(np.array(m["_mass"]), axis=0).tolist()
        roles = m["_roles"]
        out.append({"s": m["s"], "e": m["e"],
                    "role": max(set(roles), key=roles.count),
                    "conf": float(np.mean(m["_conf"])),
                    "conf_max": float(np.max(m["_conf"])),
                    "role_mass": mass, "o_mass": float(np.mean(m["_o"])),
                    "seeds": sorted(set(m["seeds"])), "n_agree": len(set(m["seeds"]))})
    out.sort(key=lambda c: (c["s"], c["e"]))
    return out


def label(window, cands):
    """Attach the metric-faithful target label to every candidate, in place."""
    for c in cands:
        best, lab = 0.0, REJECT
        for gs, ge, gr in window.role_spans:
            v = iou((c["s"], c["e"]), (gs, ge))
            if v > 0.5 and v > best:
                best, lab = v, gr
        c["label"] = lab
        c["gold_iou"] = best
    return cands


def save(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pool_rows(rows_by_seed):
    """Pool per-window candidate rows from several proposer seeds.

    rows_by_seed: list (one per seed) of lists of rows {sent_id, cands, type_post,
    aao_post}. Returns one row per window with pooled candidates and the type and
    AAO posteriors averaged over seeds. Windows keep the order of the first seed.
    """
    index = [{r["sent_id"]: r for r in rows} for rows in rows_by_seed]
    out = []
    for r0 in rows_by_seed[0]:
        sid = r0["sent_id"]
        group = [ix[sid] for ix in index if sid in ix]
        out.append({
            "sent_id": sid,
            "cands": pool([g["cands"] for g in group]),
            "type_post": np.mean([g["type_post"] for g in group], axis=0).tolist(),
            "aao_post": np.mean([np.asarray(g["aao_post"]) for g in group], axis=0).tolist(),
        })
    return out

