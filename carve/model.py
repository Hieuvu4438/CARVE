"""Stage 1 model: the CARVE-simple proposer, a word-level span tagger.

    shared encoder (DeBERTa-v3-large)
        -> first-subword pooling to word-level states
        -> SPAN head : one BIO tagger over 13 span types
                       (9 scored roles + Agent / Action / PrimaryObject / SecondaryObject)
        -> TYPE head : 4-way window event-type classifier (attention pooled)

The scored arguments of SciEvent are long, near-contiguous, mutually
non-overlapping clause spans, so argument extraction is span segmentation: a
discriminative word tagger, not a generative or mention-level extractor.

Two design decisions of the original CARVE were removed because controlled
single-factor ablations showed them to be inert: separate BIO heads for the two
span groups, and conditioning the role head on the predicted event type. The
event-type head itself is required (the metric is event-type sensitive).
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from carve.data import EVENT_TYPES, MERGED_LABELS


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        return (x * w).sum(1)


class CarveTagger(nn.Module):
    def __init__(self, model_name, n_labels=len(MERGED_LABELS), n_event_types=len(EVENT_TYPES), dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Master weights must be fp32: deberta-v3-large ships fp16 weights and
        # transformers>=5 honours that dtype, which makes AdamW produce NaN on the
        # first step. bf16 comes from autocast only.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        h = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool(h)
        self.type_head = nn.Linear(h, n_event_types)
        # Not read by the forward pass: this is the embedding of the removed
        # type-conditioning pathway, zero-initialised. It is kept (with the random
        # draw in `forward`) so that parameter initialisation and the dropout RNG
        # stream follow the reported run. Measured: with both, one epoch of the
        # released code matches the development code to within GPU
        # nondeterminism; without them the run follows a different random stream.
        self.type_emb = nn.Embedding(n_event_types, h)
        nn.init.zeros_(self.type_emb.weight)
        self.role_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_labels))

    def word_states(self, input_ids, attention_mask, word_index, word_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # gather the first sub-token of every word
        idx = word_index.unsqueeze(-1).expand(-1, -1, out.size(-1))
        w = torch.gather(out, 1, idx)
        w = w * word_mask.unsqueeze(-1)
        return self.dropout(w)

    def forward(self, input_ids, attention_mask, word_index, word_mask, training=False):
        """Returns (type_logits [B, 4], span_logits [B, W, 27])."""
        w = self.word_states(input_ids, attention_mask, word_index, word_mask)
        type_logits = self.type_head(self.pool(w, word_mask.bool()))
        if training:   # value unused; see `type_emb` above
            torch.rand_like(type_logits.argmax(-1), dtype=torch.float)
        return type_logits, self.role_head(w)
