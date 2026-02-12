from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SEMComparisonBuilder:
    """
    Builds SEM_TECHNIQUE_COMPARISON.xlsx by aligning
    REAL and synthetic aggregated SEM outputs.

    Phase A of the pipeline.
    """

    VALID_SUFFIXES = ("_mean", "_std", "_range", "_mea")

    def __init__(
        self,
        real_path: str,
        technique_paths: dict[str, str],
        cnt: str,
        out_file: str | Path,
    ):
        self.technique_paths = technique_paths
        self.cnt = cnt
        self.out_file = Path(out_file)
        self.real_path = real_path

        self.out_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_real(self) -> dict:
        file_path = Path(self.real_path) / "sem_outputs" / f"sem_results_df_core{self.cnt}.xlsx"

        if not file_path.exists():
            logger.error(f"REAL SEM file missing: {file_path}")
            return {}

        xl = pd.ExcelFile(file_path, engine="openpyxl")
        logger.info(f"REAL: {len(xl.sheet_names)} sheets found")

        sheets = {}
        for raw in xl.sheet_names:
            norm = self._normalize_sheet_name(raw)   # ❗ NO "_mean"
            sheets[norm] = xl.parse(raw)

        return sheets


    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================
    def run(self):
        logger.info("Running SEMComparisonBuilder")

        all_results = self._load_all_techniques()

        # ---- LOAD REAL EXPLICITLY ----
        all_results["REAL"] = self._load_real()
        self._write_comparison(all_results)

        logger.info(f"SEM comparison workbook written to: {self.out_file}")

    # ======================================================
    # LOADING
    # ======================================================
    def _load_all_techniques(self) -> dict:
        all_results = {}

        for tech, path in self.technique_paths.items():
            logger.info(f"Loading technique: {tech}")
            all_results[tech] = self._load_single_technique(
                tech, Path(path)
            )

        return all_results

    def _load_single_technique(self, tech: str, base_path: Path) -> dict:
        """
        Load all sheets for a single technique into a dict:
        {normalized_sheet_name: DataFrame}
        """

        file_path = base_path / "sem_outputs" / "SEM_AGGREGATED_RESULTS.xlsx"

        if not file_path.exists():
            logger.warning(f"[SKIP] Missing file for {tech}: {file_path}")
            return {}

        xl = pd.ExcelFile(file_path, engine="openpyxl")
        logger.info(f"{tech}: {len(xl.sheet_names)} sheets found")

        sheets = {}
        for raw in xl.sheet_names:
            norm = self._normalize_sheet_name(raw)
            sheets[norm] = xl.parse(raw)

        return sheets

    # ======================================================
    # CORE COMPARISON
    # ======================================================
    def _write_comparison(self, all_results: dict):

        all_sheet_names = set()
        for sheets in all_results.values():
            for name in sheets:
                if name.endswith(self.VALID_SUFFIXES):
                    all_sheet_names.add(name)

        with pd.ExcelWriter(self.out_file, engine="openpyxl") as writer:
            for sheet in sorted(all_sheet_names):
                logger.info(f"Comparing sheet: {sheet}")

                merged = self._compare_sheet(sheet, all_results)
                if merged is None:
                    logger.warning(f"[SKIP] Not enough data for {sheet}")
                    continue

                merged.to_excel(
                    writer,
                    sheet_name=sheet[:31],
                    index=False
                )

    def _compare_sheet(self, sheet_name: str, all_results: dict) -> pd.DataFrame | None:
        """
        Align one logical sheet across techniques.
        For *_range sheets:
        - REAL appears as a single reference column
        - Techniques appear as min/max bands
        """

        dfs = {}
        if sheet_name.endswith("_mean"):
            real_equiv = sheet_name.replace("_mean", "")
        elif sheet_name.endswith("_range"):
            real_equiv = sheet_name.replace("_range", "")
        else:
            real_equiv = None

        for tech, sheets in all_results.items():
            if tech == "REAL":
                if real_equiv and real_equiv in sheets:
                    dfs[tech] = sheets[real_equiv]
            else:
                if sheet_name in sheets:
                    dfs[tech] = sheets[sheet_name]

        if len(dfs) < 2:
            return None

        # Prefer REAL as base if present
        base_df = dfs.get("REAL", next(iter(dfs.values())))
        keys = self._id_cols(base_df)

        # -------------------------------
        # Pass 1: merge all techniques
        # -------------------------------
        merged = None

        for tech, df in dfs.items():
            df = df.copy()

            if sheet_name.endswith("_range"):
                if tech == "REAL":
                    metric_cols = self._numeric_cols(df)
                else:
                    metric_cols = [c for c in df.columns if c not in keys]
            else:
                metric_cols = self._numeric_cols(df)

            rename = {c: f"{tech}__{c}" for c in metric_cols}
            df = df.rename(columns=rename)

            keep = keys + list(rename.values())

            if merged is None:
                merged = df[keep]
            else:
                merged = merged.merge(
                    df[keep],
                    on=keys,
                    how="left",
                    sort=False
                )

        # -------------------------------
        # Pass 2: column ordering
        # -------------------------------
        tech_order = ["REAL"] + [t for t in dfs.keys() if t != "REAL"]
        ordered_cols = keys.copy()

        if sheet_name.endswith("_range"):
            # Collect metric roots across REAL and techniques
            metric_roots = set()

            for tech, df in dfs.items():
                if tech == "REAL":
                    metric_roots.update(self._numeric_cols(df))
                else:
                    metric_roots.update(
                        c[:-4] for c in df.columns
                        if c.endswith("_min") or c.endswith("_max")
                    )

            for r in sorted(metric_roots):
                for tech in tech_order:
                    if tech == "REAL":
                        col = f"REAL__{r}"
                        if col in merged.columns:
                            ordered_cols.append(col)
                    else:
                        for suffix in ("_min", "_max"):
                            col = f"{tech}__{r}{suffix}"
                            if col in merged.columns:
                                ordered_cols.append(col)
        else:
            base_metrics = self._numeric_cols(base_df)
            for m in base_metrics:
                for tech in tech_order:
                    col = f"{tech}__{m}"
                    if col in merged.columns:
                        ordered_cols.append(col)

        logger.info(f"{sheet_name}: techniques present = {list(dfs.keys())}")

        return merged[ordered_cols]


    # ======================================================
    # HELPERS
    # ======================================================
    @staticmethod
    def _numeric_cols(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def _id_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in SEMComparisonBuilder._numeric_cols(df)]

    @staticmethod
    def _normalize_sheet_name(name: str) -> str:
        if name.endswith("_mea"):
            return name[:-4] + "_mean"
        if name.endswith("_ran"):
            return name[:-4] + "_range"
        return name

    @staticmethod
    def _real_equivalent(sheet: str) -> str | None:
        if sheet.endswith("_range"):
            return sheet.replace("_range", "")
        if sheet.endswith("_std"):
            return None
        if sheet.endswith("_mean"):
            return sheet.replace("_mean", "")
        return sheet
