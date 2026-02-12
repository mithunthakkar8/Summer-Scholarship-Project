from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder



logger = logging.getLogger(__name__)


class PrivacyEvaluator:
    """
    Privacy evaluation for one REAL vs one SYNTHETIC dataset.

    Tier 1 (Primary, reportable):
      - Exact Match Rate (SDV)
      - Nearest Neighbor Distance Ratio (SDV)
      - Membership Inference Risk (SDV)
      - Distance to Closest Record (manual)

    Tier 2 (Secondary, appendix):
      - k-Anonymity proxy (distance-based)
      - Attribute Disclosure Risk (attack-style)

    CPU-only, deterministic, no training.
    """

    def __init__(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        technique: str,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        k_anonymity: int = 5,
        distance_metric: str = "euclidean",
    ):
        self.real_df = real_df.reset_index(drop=True)
        self.synthetic_df = synthetic_df.reset_index(drop=True)
        self.technique = technique

        self.quasi_identifiers = quasi_identifiers or []
        self.sensitive_attributes = sensitive_attributes or []
        self.k = k_anonymity
        self.distance_metric = distance_metric

        self._validate_inputs()

    # ======================================================
    # Validation
    # ======================================================

    def _validate_inputs(self):
        if list(self.real_df.columns) != list(self.synthetic_df.columns):
            raise ValueError("REAL and SYNTHETIC schemas must match exactly.")

        for col in self.quasi_identifiers + self.sensitive_attributes:
            if col not in self.real_df.columns:
                raise KeyError(f"Column not found: {col}")

        if self.k < 2:
            raise ValueError("k-anonymity proxy requires k >= 2.")

    # ======================================================
    # Shared numeric encoding (manual metrics)
    # ======================================================

    def _numeric_matrices(self):
        real_num = self.real_df.select_dtypes(include=[np.number])
        synth_num = self.synthetic_df.select_dtypes(include=[np.number])

        common_cols = real_num.columns.intersection(synth_num.columns)

        if len(common_cols) == 0:
            raise ValueError("No numeric columns available for distance metrics.")

        return real_num[common_cols].values, synth_num[common_cols].values


    # ======================================================
    # Tier 1 – Manual, SDV-independent (RECOMMENDED)
    # ======================================================

    def _exact_match_rate(self) -> float:
        real_rows = set(map(tuple, self.real_df.values))
        synth_rows = list(map(tuple, self.synthetic_df.values))
        matches = sum(r in real_rows for r in synth_rows)
        return matches / len(synth_rows)


    def _nn_distance_mean(self) -> float:
        real_X, synth_X = self._numeric_matrices()
        dists = pairwise_distances(synth_X, real_X)
        return float(dists.min(axis=1).mean())


    def _membership_inference_proxy(self) -> float:
        """
        Proxy: fraction of synthetic rows within ε of a real row.
        """
        real_X, synth_X = self._numeric_matrices()
        dists = pairwise_distances(synth_X, real_X)
        eps = np.percentile(dists, 1)
        return float((dists.min(axis=1) < eps).mean())


    # ======================================================
    # Tier 1 – Manual (required)
    # ======================================================

    def _distance_to_closest_record(self) -> Dict[str, float]:
        real_X, synth_X = self._numeric_matrices()

        dists = pairwise_distances(
            synth_X,
            real_X,
            metric=self.distance_metric,
        )

        min_dists = dists.min(axis=1)

        return {
            "mean": float(np.mean(min_dists)),
            "min": float(np.min(min_dists)),
            "p05": float(np.percentile(min_dists, 5)),
        }

    # ======================================================
    # Tier 2 – Advanced metrics
    # ======================================================

    def _k_anonymity_proxy(self) -> Dict[str, float]:
        real_X, synth_X = self._numeric_matrices()

        dists = pairwise_distances(synth_X, real_X)
        kth = np.sort(dists, axis=1)[:, self.k - 1]

        return {
            "mean_k_distance": float(np.mean(kth)),
            "min_k_distance": float(np.min(kth)),
        }

    def _attribute_disclosure(self) -> Optional[Dict[str, float]]:
        if not self.quasi_identifiers or not self.sensitive_attributes:
            return None

        enc = OneHotEncoder(handle_unknown="ignore")

        real_qi = enc.fit_transform(self.real_df[self.quasi_identifiers])
        synth_qi = enc.transform(self.synthetic_df[self.quasi_identifiers])

        dists = pairwise_distances(synth_qi, real_qi)
        nn_idx = dists.argmin(axis=1)

        risks = {}
        for attr in self.sensitive_attributes:
            risks[attr] = float(
                np.mean(
                    self.real_df.iloc[nn_idx][attr].values
                    == self.synthetic_df[attr].values
                )
            )

        return risks

    # ======================================================
    # Public API
    # ======================================================

    def run(self) -> Dict[str, object]:
        logger.info("Running privacy evaluation: %s", self.technique)

        return {
            "Technique": self.technique,
            "Tier1": {
                "ExactMatchRate": self._exact_match_rate(),
                "NNDR": self._nn_distance_mean(),
                "MembershipInference": self._membership_inference_proxy(),
                "DCR": self._distance_to_closest_record(),
            },
            "Tier2": {
                "KAnonymity": self._k_anonymity_proxy(),
                "AttributeDisclosure": self._attribute_disclosure(),
            },
        }
