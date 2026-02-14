"""
Anomaly detection framework: Level 1 (statistical) + Level 2 (ML) + ensemble voting.
Trained on AIMD normal data; tested on MLFF trajectories.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class StatisticalDetector:
    """Level 1: 3-sigma threshold detector per feature."""

    def __init__(self, n_sigma: float = 3.0):
        self.n_sigma = n_sigma
        self.thresholds: Dict[str, Dict] = {}
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, feature_names: List[str]):
        self.feature_names = feature_names
        for i, name in enumerate(feature_names):
            col = X[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                self.thresholds[name] = {'mean': 0, 'std': 1, 'upper': np.inf, 'lower': -np.inf}
                continue
            m, s = np.mean(valid), np.std(valid)
            self.thresholds[name] = {
                'mean': float(m),
                'std': float(s),
                'upper': float(m + self.n_sigma * s),
                'lower': float(m - self.n_sigma * s),
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal. Shape: (n_samples,)"""
        anomalies = np.zeros(len(X), dtype=int)
        for i, name in enumerate(self.feature_names):
            if name not in self.thresholds:
                continue
            t = self.thresholds[name]
            col = X[:, i]
            flag = (col > t['upper']) | (col < t['lower'])
            anomalies |= flag.astype(int)
        return anomalies

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return fraction of features outside thresholds per sample."""
        scores = np.zeros(len(X))
        for i, name in enumerate(self.feature_names):
            if name not in self.thresholds:
                continue
            t = self.thresholds[name]
            col = X[:, i]
            flag = ((col > t['upper']) | (col < t['lower'])).astype(float)
            scores += flag
        return scores / max(len(self.feature_names), 1)


class AnomalyDetectionFramework:
    """
    Three-level ensemble anomaly detector.

    Level 1: Statistical (3-sigma thresholds)
    Level 2a: Isolation Forest
    Level 2b: One-Class SVM
    Ensemble: majority vote (anomaly if ≥2 detectors agree)
    """

    def __init__(self, contamination: float = 0.05, n_sigma: float = 3.0):
        self.contamination = contamination
        self.n_sigma = n_sigma
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

        self.level1 = StatisticalDetector(n_sigma=n_sigma)
        self.level2_if = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
        )
        self.level2_svm = OneClassSVM(kernel='rbf', nu=contamination)
        self._fitted = False

    def fit(self, X: np.ndarray, feature_names: List[str]):
        """Train all detectors on normal (AIMD) data."""
        self.feature_names = feature_names

        # Replace NaN with column medians before scaling
        X_clean = self._impute(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.level1.fit(X_clean, feature_names)
        self.level2_if.fit(X_scaled)
        self.level2_svm.fit(X_scaled)
        self._fitted = True
        print(f"✓ Detectors trained on {len(X)} windows, {len(feature_names)} features")

    def predict(self, X: np.ndarray) -> Dict:
        """
        Returns dict with per-sample predictions and scores.
        Keys:
            'anomaly_label':  (n,) int  1=anomaly 0=normal
            'confidence':     (n,) int  0-3 (how many detectors voted anomaly)
            'l1_score':       (n,) float  fraction of features outside 3-sigma
            'l2_if_score':    (n,) float  IF anomaly score (more negative = more anomalous)
            'l2_svm_score':   (n,) float  SVM decision score (negative = anomalous)
            'l1_flag':        (n,) int
            'l2_if_flag':     (n,) int
            'l2_svm_flag':    (n,) int
        """
        assert self._fitted, "Call fit() before predict()"
        X_clean = self._impute(X)
        X_scaled = self.scaler.transform(X_clean)

        l1_flag = self.level1.predict(X_clean)
        l1_score = self.level1.score(X_clean)

        # IsolationForest: predict returns -1 for anomaly
        if_preds = self.level2_if.predict(X_scaled)
        if_scores = self.level2_if.score_samples(X_scaled)   # more negative = anomalous
        l2_if_flag = (if_preds == -1).astype(int)

        svm_preds = self.level2_svm.predict(X_scaled)
        svm_scores = self.level2_svm.decision_function(X_scaled)
        l2_svm_flag = (svm_preds == -1).astype(int)

        confidence = l1_flag + l2_if_flag + l2_svm_flag
        anomaly_label = (confidence >= 2).astype(int)

        return {
            'anomaly_label': anomaly_label,
            'confidence': confidence,
            'l1_flag': l1_flag,
            'l2_if_flag': l2_if_flag,
            'l2_svm_flag': l2_svm_flag,
            'l1_score': l1_score,
            'l2_if_score': if_scores,
            'l2_svm_score': svm_scores,
        }

    def anomaly_rate(self, X: np.ndarray) -> float:
        results = self.predict(X)
        return float(np.mean(results['anomaly_label']))

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Framework saved → {path}")

    @staticmethod
    def load(path: str) -> 'AnomalyDetectionFramework':
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _impute(X: np.ndarray) -> np.ndarray:
        """Replace NaN with column median (computed ignoring NaNs)."""
        X_out = X.copy()
        for col in range(X_out.shape[1]):
            col_vals = X_out[:, col]
            nan_mask = np.isnan(col_vals)
            if nan_mask.any():
                median = np.nanmedian(col_vals)
                X_out[nan_mask, col] = median if not np.isnan(median) else 0.0
        return X_out
