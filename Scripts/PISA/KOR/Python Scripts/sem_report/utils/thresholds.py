"""
Centralised thresholds for SEM evaluation.

All evaluator modules MUST import thresholds from here.
This ensures consistency across reliability, validity, paths, etc.
"""

# =====================================================
# RELIABILITY THRESHOLDS (PLS-SEM conventions)
# =====================================================

RELIABILITY_THRESHOLDS = {
    "alpha": 0.70,   # Cronbach's alpha
    "rhoC": 0.70,    # Composite reliability
    "rhoA": 0.70,    # rho_A
    "AVE": 0.50,     # Average variance extracted
}

# -----------------------------------------------------
# Optional: thresholds for stability interpretation
# -----------------------------------------------------

RELIABILITY_RANGE_THRESHOLDS = {
    "alpha": 0.30,
    "rhoC": 0.30,
    "rhoA": 0.30,
    "AVE": 0.10,
}

# Heterotrait-Monotrait ratio maximum for discriminant validity

HTMT_THRESHOLD = 0.85  


CBSEM_GLOBAL_FIT_THRESHOLDS = {
    "RMSEA": {"direction": "lt", "cutoff": 0.08},
    "SRMR":  {"direction": "lt", "cutoff": 0.08},
    "CFI":   {"direction": "gt", "cutoff": 0.90},
    "TLI":   {"direction": "gt", "cutoff": 0.90},
}
