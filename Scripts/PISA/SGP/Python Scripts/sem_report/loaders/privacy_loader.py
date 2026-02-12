from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class PrivacyLoader:
    """
    Loads privacy evaluation outputs written to disk.

    Scope:
      - Mean privacy metrics
      - Stability / range metrics (if present)
      - Delta vs REAL (if present)

    This loader is intentionally:
      - CSV-based
      - Model-agnostic
      - Independent of SEMComparisonLoader
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.privacy_dir = self.out_dir / "tables" / "008_privacy"

        logger.info("Initializing PrivacyLoader")
        logger.debug(f"Privacy tables dir: {self.privacy_dir}")

        if not self.privacy_dir.exists():
            logger.error(f"Privacy tables directory not found: {self.privacy_dir}")
            raise FileNotFoundError(
                f"Privacy tables directory not found: {self.privacy_dir}"
            )

    # --------------------------------------------------
    # Core loaders
    # --------------------------------------------------

    def mean(self) -> pd.DataFrame:
        """
        Load mean privacy metrics (wide format).
        """
        path = self.privacy_dir / "privacy_mean.csv"

        logger.info("Loading privacy mean table")
        if not path.exists():
            raise FileNotFoundError(f"Missing privacy mean table: {path}")

        return pd.read_csv(path)

    def delta_vs_real(self) -> pd.DataFrame | None:
        """
        Load delta vs REAL privacy table (optional).
        """
        path = self.privacy_dir / "privacy_delta_vs_real.csv"

        if not path.exists():
            logger.info("No privacy delta_vs_real table found")
            return None

        return pd.read_csv(path)

    def stability(self) -> pd.DataFrame | None:
        """
        Load privacy stability / range table (optional).
        """
        path = self.privacy_dir / "privacy_stability.csv"

        if not path.exists():
            logger.info("No privacy stability table found")
            return None

        return pd.read_csv(path)
