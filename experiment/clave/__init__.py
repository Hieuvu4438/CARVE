"""Experiment-only overlay for CLAVE.

Only modules explicitly placed here differ from the released package. All
other ``clave.*`` imports resolve to the read-only released source tree.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "clave"))
