"""Equitable post-HCT survival prediction.

Reference implementation for:

    S. Kanakala, A. Chetan Krishna Sai, A. Badhri Yadav, A. Arisenapally and
    M. V. S. Surya, "Post-Hematopoietic Cell Transplantation Survival
    Predictions using Ensemble Machine Learning Models," 2026 7th
    International Conference on Inventive Research in Computing Applications
    (ICIRCA), Coimbatore, India, 2026.
    doi: 10.1109/ICIRCA69024.2026.11570591
"""

__version__ = "1.0.0"

from hct_survival.config import Config, CVConfig, Paths

__all__ = ["Config", "CVConfig", "Paths", "__version__"]
