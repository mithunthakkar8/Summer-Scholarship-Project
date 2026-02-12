from pathlib import Path
import pandas as pd
import logging

from sem_report.utils.export_config import ExportConfig

logger = logging.getLogger(__name__)

FORMAT_CSS = """
<style>
/* =====================================================
   APA 7 TABLE FORMAT (MINIMAL RULES)
   ===================================================== */

.sem-table {
  border-collapse: collapse;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 12pt;
  color: #000;
  margin: 12px 0;

  /* APA: only top & bottom rules */
  border-top: 2px solid #000;
  border-bottom: 2px solid #000;
}

/* -----------------------------------------------------
   HEADER
   ----------------------------------------------------- */
.sem-table thead th {
  padding: 6px 8px;
  text-align: center;
  font-weight: 600;

  /* APA: header separator rule */
  border-bottom: 1px solid #000;

  /* APA: NEVER vertical rules */
  border-left: none;
  border-right: none;
}

/* -----------------------------------------------------
   BODY
   ----------------------------------------------------- */
.sem-table td {
  padding: 6px 8px;
  text-align: center;
  vertical-align: middle;

  /* APA: no inner gridlines */
  border: none;
  white-space: nowrap;
}

/* First column left-aligned (row labels) */
.sem-table th:first-child,
.sem-table td:first-child {
  text-align: left;
}

/* Remove zebra striping / shading */
.sem-table tbody tr {
  background: transparent;
}

/* -----------------------------------------------------
   TABLE NOTES
   ----------------------------------------------------- */
.apa-table-note {
  font-size: 11pt;
  margin-top: 6px;
}

/* -----------------------------------------------------
   TABLE NUMBER & TITLE
   ----------------------------------------------------- */
.apa-table-number {
  margin-top: 12px;
  font-size: 12pt;
  font-weight: 600;
}

.apa-table-title {
  margin-bottom: 6px;
  font-size: 12pt;
  font-style: italic;
}

/* -----------------------------------------------------
   SECTION HEADERS (SPAN FULL WIDTH)
   ----------------------------------------------------- */
.sem-table .section-header th {
  text-align: center;
  font-weight: 600;
  padding: 6px 0;
  border-top: 1px solid #000;
  border-bottom: 1px solid #000;
}

</style>
"""


MATHJAX = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""

def inject_section_headers(html: str, sections: dict[str, list[str]]) -> str:
    """
    Inject APA-style section header rows with colspan into HTML table.
    
    sections = {
        "DISTRIBUTION FIDELITY": ["Mean |Δμ| ↓", "Mean |Δσ| ↓", ...],
        "STRUCTURAL FIDELITY": [...],
        ...
    }
    """
    for section, first_metrics in sections.items():
        for metric in first_metrics:
            marker = f"<td>{metric}</td>"
            header_row = (
                "<tr class=\"section-header\">"
                f"<th colspan=\"{html.count('<th>')}\">— {section} —</th>"
                "</tr>"
            )
            html = html.replace(marker, header_row + marker, 1)
            break
    return html


def export_table(
    df: pd.DataFrame,
    path: Path,
    config: ExportConfig,
    *,
    table_number: int | None = None,
    title: str | None = None,
    note: str | None = None,
    index: bool = False,
    float_format: str = "%.3f",
):

    """
    Export a DataFrame according to the global export policy.
    """

    # --------------------------------------------------
    # Enforce no index export (GLOBAL RULE)
    # --------------------------------------------------
    if df.index.name is not None or not df.index.equals(pd.RangeIndex(len(df))):
        logger.debug("Resetting DataFrame index before export")
        df = df.reset_index(drop=True)

    index = False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config.export_csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=index, float_format=float_format)
        logger.debug(f"CSV written: {csv_path}")

    # --------------------------------------------------
    # APA 7 LaTeX export (optional)
    # --------------------------------------------------
    if getattr(config, "export_latex", False):
        tex_path = path.with_suffix(".tex")

        latex_table = df.to_latex(
            index=index,
            float_format=float_format,
            escape=False,
            column_format="l" + "c" * (len(df.columns) - 1),
            caption=title,
            label=f"tab:{path.stem}",
        )

        if table_number is not None:
            latex_table = latex_table.replace(
                "\\begin{table}",
                f"\\begin{{table}}[ht]\n\\caption*{{\\textbf{{Table {table_number}}}}}",
            )

        if note:
            latex_table = latex_table.replace(
                "\\end{tabular}",
                "\\end{tabular}\n\\\\\n\\textit{Note.} " + note,
            )

        tex_path.write_text(latex_table, encoding="utf-8")


    if config.export_html:
        html_path = path.with_suffix(".html")
        # --------------------------------------------------
        # APA 7 table number & title
        # --------------------------------------------------
        caption_html = ""

        if table_number is not None:
            caption_html += (
                f'<div class="apa-table-number">'
                f'<strong>Table {table_number}</strong>'
                f'</div>'
            )

        if title:
            caption_html += (
                f'<div class="apa-table-title">'
                f'<em>{title}</em>'
                f'</div>'
            )

        html = df.to_html(
            index=index,
            classes="sem-table",
            escape=False
        )

        # --------------------------------------------------
        # APA 7 table note
        # --------------------------------------------------
        note_html = ""

        if note:
            note_html = (
                f'<div class="apa-table-note">'
                f'<em>Note.</em> {note}'
                f'</div>'
            )
        
        html = (
            FORMAT_CSS
            + MATHJAX
            + caption_html
            + html
            + note_html
        )


        html_path.write_text(html, encoding="utf-8")
        logger.debug(f"HTML written: {html_path}")


    

