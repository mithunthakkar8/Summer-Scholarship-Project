from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SEMComparisonLoader:
    """
    Responsible ONLY for loading and aligning SEM comparison results
    from the merged comparison workbook.

    No metrics. No plots. No interpretation.
    """

    VALID_SUFFIXES = ("_mean", "_mea", "_range", "_std")

    def __init__(self, comparison_xlsx: str | Path):
        self.path = Path(comparison_xlsx)

        logger.info("Initializing SEMComparisonLoader")
        logger.debug(f"Comparison workbook path: {self.path}")

        if not self.path.exists():
            logger.error(f"Comparison workbook not found: {self.path}")
            raise FileNotFoundError(f"Comparison workbook not found: {self.path}")

        self.xl = pd.ExcelFile(self.path, engine="openpyxl")
        self.sheet_names = self.xl.sheet_names

        logger.info(
            f"Workbook loaded successfully "
            f"(sheets={len(self.sheet_names)})"
        )
        logger.debug(f"Available sheets: {self.sheet_names}")

    # ------------------------------------------------------------------
    # Core sheet resolution
    # ------------------------------------------------------------------
    def find_mean_sheet(self, prefix: str) -> str:
        """
        Resolve MEAN sheet robust to Excel truncation.
        """
        logger.debug(f"Resolving MEAN sheet for prefix='{prefix}'")

        candidates = [
            s for s in self.sheet_names
            if s.startswith(prefix) and s.endswith(("_mean", "_mea"))
        ]

        if len(candidates) == 0:
            logger.error(f"No MEAN sheet found for prefix='{prefix}'")
            raise ValueError(
                f"No MEAN sheet found for prefix '{prefix}'. "
                f"Available sheets: {self.sheet_names}"
            )

        if len(candidates) > 1:
            logger.error(
                f"Ambiguous MEAN sheets for prefix='{prefix}': {candidates}"
            )
            raise ValueError(
                f"Ambiguous MEAN sheets for '{prefix}': {candidates}"
            )

        logger.debug(f"Resolved MEAN sheet: {candidates[0]}")
        return candidates[0]

    def find_range_sheet(self, prefix: str) -> str | None:
        logger.debug(f"Resolving RANGE sheet for prefix='{prefix}'")

        candidates = [
            s for s in self.sheet_names
            if s.startswith(prefix) and s.endswith("_range")
        ]

        if len(candidates) == 1:
            logger.debug(f"Resolved RANGE sheet: {candidates[0]}")
            return candidates[0]

        if len(candidates) > 1:
            logger.warning(
                f"Multiple RANGE sheets for prefix='{prefix}': {candidates}"
            )

        logger.info(f"No RANGE sheet found for prefix='{prefix}'")
        return None

    # ------------------------------------------------------------------
    # Generic loader
    # ------------------------------------------------------------------
    def load_sheet(
        self,
        prefix: str,
        stat: str = "mean",
        coerce_numeric: bool = True
    ) -> pd.DataFrame:
        """
        Load a logical SEM result sheet.

        stat ∈ {"mean", "range", "std"}
        """
        logger.info(f"Loading sheet: prefix='{prefix}', stat='{stat}'")

        if stat == "mean":
            sheet = self.find_mean_sheet(prefix)
        elif stat == "range":
            sheet = self.find_range_sheet(prefix)
            if sheet is None:
                logger.error(f"No RANGE sheet for prefix='{prefix}'")
                raise ValueError(f"No RANGE sheet for prefix '{prefix}'")
        elif stat == "std":
            sheet = f"{prefix}_std"
            if sheet not in self.sheet_names:
                logger.error(f"No STD sheet found: {sheet}")
                raise ValueError(f"No STD sheet for prefix '{prefix}'")
        else:
            logger.error(f"Invalid stat='{stat}' requested")
            raise ValueError(f"Invalid stat='{stat}'")

        logger.debug(f"Parsing Excel sheet: {sheet}")
        df = self.xl.parse(sheet)

        logger.info(
            f"Loaded sheet '{sheet}' "
            f"(rows={df.shape[0]}, cols={df.shape[1]})"
        )

        # Coerce numeric ONLY where appropriate
        if coerce_numeric and stat in {"mean", "std"}:
            df = self._coerce_numeric(df)

        # Optional sanity check for new RANGE format
        if stat == "range":
            range_cols = [c for c in df.columns if c.endswith("_min") or c.endswith("_max")]
            if not range_cols:
                logger.warning(
                    f"RANGE sheet '{sheet}' has no *_min / *_max columns"
                )

        return df


    # ------------------------------------------------------------------
    # Domain-specific accessors
    # ------------------------------------------------------------------

    def latent_correlations_pls(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing PLS latent correlations for CNT={cnt}")
        return self.load_sheet(f"pls_sem_correlations_{cnt}")

    def latent_correlations_cb(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing CB-SEM latent correlations for CNT={cnt}")
        return self.load_sheet(f"sem_cb_correlations_{cnt}")

    def fornell_larcker(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing Fornell–Larcker matrix for CNT={cnt}")
        return self.load_sheet(f"pls_sem_fornell_larcker_{cnt}")

    def htmt(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing HTMT matrix for CNT={cnt}")
        return self.load_sheet(f"pls_sem_htmt_{cnt}")

    def loadings(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing loadings for CNT={cnt}")
        return self.load_sheet(f"pls_sem_loadings_R2_{cnt}")

    def standardized_paths(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing standardized paths for CNT={cnt}")
        return self.load_sheet(f"sem_full_std_paths_{cnt}")

    def standardized_paths_range(self, cnt: str) -> pd.DataFrame | None:
        logger.debug(f"Accessing standardized path ranges for CNT={cnt}")
        try:
            return self.load_sheet(
                f"sem_full_std_paths_{cnt}",
                stat="range"
            )
        except ValueError:
            logger.info(f"No path RANGE sheet for CNT={cnt}")
            return None

    def indirect_effects(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing indirect effects for CNT={cnt}")
        return self.load_sheet(f"pls_sem_full_indirect_{cnt}")

    def total_effects(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing total effects for CNT={cnt}")
        return self.load_sheet(f"sem_full_total_effects_{cnt}")

    def reliability(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing reliability metrics for CNT={cnt}")
        return self.load_sheet(f"pls_sem_reliability_{cnt}")

    def reliability_range(self, cnt: str) -> pd.DataFrame | None:
        logger.debug(f"Accessing reliability RANGE for CNT={cnt}")
        try:
            return self.load_sheet(
                f"pls_sem_reliability_{cnt}",
                stat="range"
            )
        except ValueError:
            logger.info(f"No reliability RANGE sheet for CNT={cnt}")
            return None

    def cb_global_fit(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing CB-SEM global fit for CNT={cnt}")
        return self.load_sheet(f"sem_cb_fit_measures_{cnt}")

    def cb_global_fit_range(self, cnt: str) -> pd.DataFrame | None:
        logger.debug(f"Accessing CB-SEM global fit RANGE for CNT={cnt}")
        try:
            return self.load_sheet(
                f"sem_cb_fit_measures_{cnt}",
                stat="range"
            )
        except ValueError:
            logger.info(f"No CB-SEM global fit RANGE for CNT={cnt}")
            return None

    def r_squared(self, cnt: str) -> pd.DataFrame:
        logger.debug(f"Accessing R² for CNT={cnt}")
        return self.load_sheet(f"sem_cb_rsquare_{cnt}")

    def covariate_corr_mean(self, cnt: str) -> pd.DataFrame:
        return self.load_sheet(
            f"covariate_corr_{cnt}_mean"
        )

    def covariate_corr_range(self, cnt: str) -> pd.DataFrame:
        return self.load_sheet(
            f"covariate_corr_{cnt}_range", stat="range"
        )


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all REAL__* and TECH__* columns to numeric safely.
        """
        out = df.copy()
        coerced = 0

        for c in out.columns:
            if "__" in c:
                before_na = out[c].isna().sum()
                out[c] = pd.to_numeric(out[c], errors="coerce")
                after_na = out[c].isna().sum()
                coerced += (after_na - before_na)

        if coerced > 0:
            logger.debug(
                f"Numeric coercion introduced {coerced} additional NA values"
            )

        return out


    