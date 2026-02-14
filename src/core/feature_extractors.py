"""
Feature extraction for AIMD trajectories.
All features are aggregated over atoms, so variable atom counts are handled naturally.
Features extracted on sliding windows of frames.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WindowConfig:
    window_size: int = 50
    stride: int = 10


def sliding_windows(n_frames: int, config: WindowConfig) -> List[Tuple[int, int]]:
    """Return list of (start, end) index pairs for sliding windows."""
    windows = []
    start = 0
    while start + config.window_size <= n_frames:
        windows.append((start, start + config.window_size))
        start += config.stride
    return windows


class FeatureExtractor:
    """
    Extract features from a window of trajectory frames.
    All features aggregate over atoms so variable n_atoms is handled transparently.
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()
        self.feature_names: List[str] = []  # populated on first extract

    def extract_window(
        self,
        coords: np.ndarray,   # (window_size, n_atoms, 3)
        energies: Optional[np.ndarray] = None,  # (window_size,)
    ) -> Dict[str, float]:
        """Extract all features from one window. Returns flat dict."""
        feats = {}
        feats.update(self._displacement_stats(coords))
        feats.update(self._dynamics_features(coords))
        feats.update(self._frequency_features(coords))
        feats.update(self._msd_features(coords))
        if energies is not None and not np.all(np.isnan(energies)):
            feats.update(self._energy_features(energies))
        else:
            feats.update({
                'energy_mean': np.nan,
                'energy_std': np.nan,
                'energy_trend': np.nan,
            })
        return feats

    def extract_all_windows(
        self,
        coords: np.ndarray,          # (n_frames, n_atoms, 3)
        energies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract features for every sliding window.
        Returns:
            feature_matrix: (n_windows, n_features) float32 array
            window_indices: list of (start, end) tuples
        """
        windows = sliding_windows(coords.shape[0], self.config)
        if not windows:
            raise ValueError(
                f"Trajectory too short ({coords.shape[0]} frames) for "
                f"window size {self.config.window_size}"
            )

        all_feats = []
        for start, end in windows:
            w_coords = coords[start:end]
            w_energies = energies[start:end] if energies is not None else None
            feats = self.extract_window(w_coords, w_energies)
            all_feats.append(feats)

        if not self.feature_names:
            self.feature_names = sorted(all_feats[0].keys())

        matrix = np.array(
            [[f.get(k, np.nan) for k in self.feature_names] for f in all_feats],
            dtype=np.float32,
        )
        return matrix, windows

    # ------------------------------------------------------------------ #
    #  Feature groups
    # ------------------------------------------------------------------ #

    def _displacement_stats(self, coords: np.ndarray) -> Dict[str, float]:
        """Statistical features of frame-to-frame per-atom displacements."""
        # displacements: (T-1, n_atoms, 3)
        disp = np.diff(coords, axis=0)
        # scalar per-atom per-step displacement magnitude
        disp_mag = np.linalg.norm(disp, axis=-1).ravel()  # flatten all

        return {
            'disp_mean': float(np.mean(disp_mag)),
            'disp_std': float(np.std(disp_mag)),
            'disp_skew': float(stats.skew(disp_mag)),
            'disp_kurtosis': float(stats.kurtosis(disp_mag)),
            'disp_max': float(np.max(disp_mag)),
            'disp_median': float(np.median(disp_mag)),
            'disp_p95': float(np.percentile(disp_mag, 95)),
        }

    def _dynamics_features(self, coords: np.ndarray) -> Dict[str, float]:
        """Velocity-based and peak-factor features."""
        disp = np.diff(coords, axis=0)               # (T-1, n_atoms, 3)
        disp_mag = np.linalg.norm(disp, axis=-1)     # (T-1, n_atoms)

        # RMS velocity (atom-averaged, then window-averaged)
        velocities_sq = disp_mag ** 2                 # (T-1, n_atoms)
        rms_velocity = float(np.sqrt(np.mean(velocities_sq)))

        # Crest factor: peak / RMS
        rms_disp = float(np.sqrt(np.mean(disp_mag ** 2)))
        peak_disp = float(np.max(disp_mag))
        crest_factor = peak_disp / (rms_disp + 1e-10)

        # Impulse factor: peak / mean
        mean_disp = float(np.mean(disp_mag))
        impulse_factor = peak_disp / (mean_disp + 1e-10)

        # Frame-to-frame variance of mean position (global drift metric)
        mean_pos_per_frame = np.mean(coords, axis=1)  # (T, 3)
        frame_var = float(np.mean(np.var(mean_pos_per_frame, axis=0)))

        # Anisotropy: ratio max/min axis variance
        per_axis_var = np.var(np.diff(coords, axis=0).reshape(-1, 3), axis=0)  # (3,)
        anisotropy = float(np.max(per_axis_var) / (np.min(per_axis_var) + 1e-10))

        return {
            'rms_velocity': rms_velocity,
            'crest_factor': crest_factor,
            'impulse_factor': impulse_factor,
            'frame_variance': frame_var,
            'anisotropy': anisotropy,
        }

    def _frequency_features(self, coords: np.ndarray) -> Dict[str, float]:
        """Frequency-domain features via FFT of mean displacement signal."""
        # Average atom position over time → (T, 3)
        mean_pos = np.mean(coords, axis=1)
        # Displacement signal: diff of mean position
        disp_signal = np.diff(mean_pos, axis=0)  # (T-1, 3)
        # Combine all axes
        combined = np.sqrt(np.sum(disp_signal ** 2, axis=1))  # (T-1,)

        fft_vals = np.abs(np.fft.rfft(combined - combined.mean()))
        power = fft_vals ** 2

        total_power = np.sum(power) + 1e-10
        norm_power = power / total_power

        # Dominant frequency index (skip DC)
        dominant_idx = int(np.argmax(fft_vals[1:]) + 1)
        dominant_freq = float(dominant_idx / len(combined))

        # Spectral entropy
        spectral_entropy = float(-np.sum(norm_power * np.log(norm_power + 1e-10)))

        # Spectral peak ratio: top-3 peaks / total
        sorted_power = np.sort(power)[::-1]
        peak_ratio = float(np.sum(sorted_power[:3]) / total_power)

        return {
            'dominant_freq': dominant_freq,
            'spectral_entropy': spectral_entropy,
            'spectral_peak_ratio': peak_ratio,
        }

    def _msd_features(self, coords: np.ndarray) -> Dict[str, float]:
        """Mean squared displacement relative to first frame."""
        ref = coords[0]  # (n_atoms, 3)
        # MSD per atom per frame
        diff = coords - ref[np.newaxis, :, :]  # (T, n_atoms, 3)
        sq_dist = np.sum(diff ** 2, axis=-1)   # (T, n_atoms)
        msd_per_frame = np.mean(sq_dist, axis=1)  # (T,) mean over atoms

        return {
            'msd_mean': float(np.mean(msd_per_frame)),
            'msd_std': float(np.std(msd_per_frame)),
            'msd_final': float(msd_per_frame[-1]),
            'msd_slope': float(
                np.polyfit(np.arange(len(msd_per_frame)), msd_per_frame, 1)[0]
            ),
        }

    def _energy_features(self, energies: np.ndarray) -> Dict[str, float]:
        """Features from frame energies."""
        valid = energies[~np.isnan(energies)]
        if len(valid) < 2:
            return {'energy_mean': np.nan, 'energy_std': np.nan, 'energy_trend': np.nan}

        trend = float(np.polyfit(np.arange(len(valid)), valid, 1)[0])
        return {
            'energy_mean': float(np.mean(valid)),
            'energy_std': float(np.std(valid)),
            'energy_trend': trend,
        }
