"""CLAVE: Cross-Fitted Clause-Level Argument Verification for scientific event extraction.

Stage 1  clave.model / clave.train / clave.decode   CARVE-simple span tagger (proposer)
Stage 2  clave.candidates / clave.verifier          out-of-fold candidates + HONE verifier
Stage 3  clave.select                               frozen decoding rule
"""

__version__ = "2.0.0"
