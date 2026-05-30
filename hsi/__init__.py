"""England Bat Hedgerow Suitability Index (HSI) decision-support tool.

A transparent, weighted multi-criteria implementation of the WSP / Sarah Proctor
(HyNet) hedgerow suitability assessment (7 suitability indices, SI1-SI7), with a
separate landscape-context layer, designed for remote/desk-based screening of
England hedgerows to prioritise static bat surveys.

The heavy GIS feature extraction lives in :mod:`hsi.pipeline` (``run_features``);
the cheap, weight-dependent scoring lives in :mod:`hsi.score` (``apply_scoring``)
so the UI can re-weight results without recomputing GIS.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
