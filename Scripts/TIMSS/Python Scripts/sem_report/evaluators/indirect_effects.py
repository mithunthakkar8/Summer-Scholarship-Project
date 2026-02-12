import logging
import pandas as pd
from sem_report.utils.exporter import export_table

logger = logging.getLogger(__name__)


class IndirectEffectsEvaluator:
    """
    Mediation / indirect effect fidelity evaluation.
    Mirrors StructuralPathEvaluator but for indirect paths.
    """

    def __init__(self, loader, cnt, techniques, out_dir, export_config):
        self.loader = loader
        self.cnt = cnt
        self.techniques = techniques
        self.out_dir = out_dir
        self.export_config = export_config

        self.tables_dir = self.out_dir / "tables" / "007_indirect_effects"
        self.appendix_dir = self.out_dir / "appendix" / "007_indirect_effects"

        for d in [self.tables_dir, self.appendix_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("Loading indirect effects sheet via loader")

        # ✅ CORRECT (uses your loader API)
        df = self.loader.indirect_effects(self.cnt)

        results = []

        for _, row in df.iterrows():

            iv, med, dv = row["IV"], row["Mediator"], row["DV"]
            real_b = row["REAL__Indirect_B"]

            for tech in self.techniques:
                syn_b = row[f"{tech}__Indirect_B"]

                results.append({
                    "IV": iv,
                    "Mediator": med,
                    "DV": dv,
                    "Technique": tech,
                    "|Δβ|": abs(syn_b - real_b),
                    "Sig Match": (
                        (row["REAL__p_value"] < 0.05)
                        == (row[f"{tech}__p_value"] < 0.05)
                    )
                })


        out_df = pd.DataFrame(results)

        export_table(
            out_df,
            self.tables_dir / "table_indirect_effects_comparison",
            config=self.export_config,
            table_number=7,   # choose next available number
            title="Indirect (Mediation) Effects Relative to Real Data",
            note=(
                "Absolute error computed as |synthetic − real|. "
                "Significance agreement indicates whether both datasets agree on p < .05."
            ),
            index=False,
        )

