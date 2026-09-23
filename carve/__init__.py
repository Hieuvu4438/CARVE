"""CARVE-simple + HONE: propose-then-verify argument extraction for SciEvent.

Stage 1  carve.model / carve.train / carve.decode   CARVE-simple span tagger (proposer)
Stage 2  carve.candidates / carve.verifier          out-of-fold candidates + HONE verifier
Stage 3  carve.select                               frozen decoding rule
"""

__version__ = "2.0.0"
