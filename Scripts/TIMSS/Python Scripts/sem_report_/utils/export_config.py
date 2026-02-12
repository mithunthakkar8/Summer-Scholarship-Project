from enum import Enum

class ExportFormat(str, Enum):
    CSV = "csv"
    HTML = "html"
    BOTH = "both"


class ExportConfig:
    """
    Global export policy for the SEM report.
    """

    def __init__(self, fmt: ExportFormat = ExportFormat.CSV):
        self.format = fmt

    @property
    def export_csv(self) -> bool:
        return self.format in (ExportFormat.CSV, ExportFormat.BOTH)

    @property
    def export_html(self) -> bool:
        return self.format in (ExportFormat.HTML, ExportFormat.BOTH)
