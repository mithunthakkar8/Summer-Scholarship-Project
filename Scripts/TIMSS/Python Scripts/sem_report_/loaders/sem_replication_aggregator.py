from pathlib import Path
import glob
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SEMReplicationAggregator:
    """
    Aggregates replicate-level SEM Excel outputs into
    SEM_AGGREGATED_RESULTS.xlsx per technique.

    Output: mean / std / range sheets.
    """

    def __init__(
        self,
        technique_paths: dict,
        cnt: str,
        sheets: list[str],
    ):
        self.technique_paths = technique_paths
        self.cnt = cnt
        self.sheets = sheets

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(self):
        for tech, input_dir in self.technique_paths.items():

            logger.info("=" * 60)
            logger.info(f"Aggregating SEM replicates: {tech}")
            logger.info(f"Directory: {input_dir}")

            self._aggregate_technique(tech, Path(input_dir))


    # --------------------------------------------------
    # Core logic
    # --------------------------------------------------
    def _aggregate_technique(self, tech: str, input_dir: Path):

        if not input_dir.exists():
            logger.warning(f"[SKIP] Directory not found: {input_dir}")
            return

        xlsx_files = [
            f for f in glob.glob(str(input_dir / "*.xlsx"))
            if not Path(f).name.startswith("SEM_AGGREGATED_RESULTS")
        ]

        if len(xlsx_files) < 2:
            logger.warning(
                f"[SKIP] {tech}: <2 replicate files found"
            )
            return

        out_file = input_dir / "SEM_AGGREGATED_RESULTS.xlsx"
        writer = pd.ExcelWriter(out_file, engine="xlsxwriter")

        for sheet in self.sheets:
            self._aggregate_sheet(sheet, xlsx_files, writer)

        writer.close()
        logger.info(f"[DONE] Written: {out_file}")

    # --------------------------------------------------
    # Sheet-level aggregation
    # --------------------------------------------------
    def _aggregate_sheet(self, sheet, xlsx_files, writer):

        frames = []

        for f in xlsx_files:
            try:
                xl = pd.ExcelFile(f, engine="openpyxl")
                if sheet not in xl.sheet_names:
                    continue

                df = xl.parse(sheet)
                df["_source_file"] = Path(f).name
                frames.append(df)

            except Exception as e:
                raise RuntimeError(f"Error reading {f}: {e}")

        if not frames:
            logger.warning(f"[SKIP] Sheet not found: {sheet}")
            return

        full_df = pd.concat(frames, ignore_index=True)

        group_cols = self._id_cols(full_df)
        group_cols = [c for c in group_cols if c != "_source_file"]

        num_cols = self._numeric_cols(full_df)

        self._write_mean_std_range(
            full_df,
            group_cols,
            num_cols,
            sheet,
            writer
        )

    # --------------------------------------------------
    # Aggregation helpers
    # --------------------------------------------------
    @staticmethod
    def _numeric_cols(df):
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def _id_cols(df):
        return [c for c in df.columns if c not in SEMReplicationAggregator._numeric_cols(df)]

    @staticmethod
    def _write_mean_std_range(df, group_cols, num_cols, sheet, writer):

        grouped = df.groupby(group_cols, sort=False)[num_cols]
        original_num_order = [c for c in df.columns if c in num_cols]

        # ---- MEAN ----
        mean_df = grouped.mean().reset_index()
        mean_df = mean_df[group_cols + original_num_order]
        mean_df.to_excel(writer, sheet_name=f"{sheet}_mean"[:31], index=False)

        # ---- STD ----
        std_df = grouped.std().reset_index()
        std_df = std_df[group_cols + original_num_order]
        std_df.to_excel(writer, sheet_name=f"{sheet}_std"[:31], index=False)

        # ---- RANGE (separate min / max columns) ----
        min_df = grouped.min().reset_index()
        max_df = grouped.max().reset_index()

        # Rename numeric columns
        min_df = min_df.rename(
            columns={c: f"{c}_min" for c in original_num_order}
        )
        max_df = max_df.rename(
            columns={c: f"{c}_max" for c in original_num_order}
        )

        # Merge min and max side-by-side
        range_df = min_df.merge(
            max_df,
            on=group_cols,
            how="left"
        )

        # Optional: round numeric columns
        for c in range_df.columns:
            if c.endswith("_min") or c.endswith("_max"):
                range_df[c] = range_df[c].round(2)

        # Column order: id cols + (var_min, var_max) pairs
        ordered_range_cols = []
        for c in original_num_order:
            ordered_range_cols.extend([f"{c}_min", f"{c}_max"])

        range_df = range_df[group_cols + ordered_range_cols]

        range_df.to_excel(
            writer,
            sheet_name=f"{sheet}_range"[:31],
            index=False
        )
