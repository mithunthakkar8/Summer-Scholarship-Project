from pathlib import Path
import pandas as pd
import logging

from sem_report.utils.export_config import ExportConfig

logger = logging.getLogger(__name__)

FORMAT_CSS = """
<style>
.sem-table {
  border-collapse: collapse;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  border: 2px solid #444;              /* strong outer border */
  margin: 12px 0;
}

/* Header cells */
.sem-table th {
  background: #f5f7fa;
  padding: 8px 10px;
  border: 1px solid #666;              /* solid header grid */
  border-bottom: 2px solid #444;       /* strong header separator */
  text-align: center;
  font-weight: 600;
}

/* Body cells */
.sem-table td {
  padding: 8px 10px;
  border: 1px solid #999;              /* solid inner grid */
  text-align: center;
  vertical-align: middle;
  white-space: nowrap;
}

/* Zebra striping for readability */
.sem-table tbody tr:nth-child(even) {
  background: #fafafa;
}

/* Emphasise first column (Path / Technique) */
.sem-table td:first-child,
.sem-table th:first-child {
  text-align: left;
  font-weight: 500;
}

/* Optional: hover highlight (harmless in PDF) */
.sem-table tbody tr:hover {
  background: #f0f4ff;
}

/* Significance colouring (optional, subtle) */
.sem-table td:contains("ns") {
  color: #666;
}

/* HTMT pass/fail retained */
.htmt-pass {
  color: #1a7f37;
  font-weight: 600;
}

.htmt-fail {
  color: #b42318;
  font-weight: 600;
}
</style>
"""


def export_table(
    df: pd.DataFrame,
    path: Path,
    config: ExportConfig,
    index: bool = False,
    float_format: str = "%.3f",
):
    """
    Export a DataFrame according to the global export policy.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config.export_csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=index, float_format=float_format)
        logger.debug(f"CSV written: {csv_path}")

    if config.export_html:
        html_path = path.with_suffix(".html")
        html = df.to_html(
            index=index,
            classes="sem-table",
            escape=False
        )

        html = FORMAT_CSS + html

        html_path.write_text(html, encoding="utf-8")
        logger.debug(f"HTML written: {html_path}")

    

