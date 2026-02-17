"""
Feature extraction for AIMD trajectories.
All features are aggregated over atoms, so variable atom counts are handled naturally.
Features extracted on sliding windows of frames.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Isolated atom DFT reference energies (eV) from data/raw/reference/isolated_atom.xyz
# Used to compute per-atom atomization energy:
#   E_atm/atom = (E_tot - Σ_i N_i * E0_i) / N_total
ISOLATED_ATOM_ENERGIES: Dict[str, float] = {
    'Cr': -2270.518789079322,
    'Sb': -1908.633351763048,
    'Te': -2369.398259284361,
}


@dataclass
class WindowConfig:
    window_size: int = 50
    stride: int = 10


def compute_file_energy_shift(
    energies: np.ndarray,
    species: List[str],
    target_atm_per_atom: float,
) -> float:
    """
    Option B empirical energy shift (MACE/NequIP approach).

    Computes a constant per-frame correction (eV) so that this file's mean
    atomization energy matches `target_atm_per_atom` — the reference value
    computed from all DFT-compatible training files.

    Derivation
    ----------
    We want:  (E_corrected - ref_sum) / N  =  target_atm_per_atom
    So:       E_corrected  =  target_atm_per_atom * N + ref_sum
    Shift/frame = E_corrected_mean - E_raw_mean
               = (target_atm_per_atom * N + ref_sum) - mean(E_raw)

    Properties
    ----------
    • energy_mean  → target_atm_per_atom  (systematic offset removed)
    • energy_std   → unchanged            (frame-to-frame fluctuations preserved)
    • energy_trend → unchanged            (drift signal preserved)

    Returns 0.0 if the file is already compatible (no shift needed).
    """
    from collections import Counter
    n = len(species)
    if n == 0:
        return 0.0
    valid = energies[~np.isnan(energies)]
    if len(valid) == 0:
        return 0.0

    counts: Dict[str, int] = Counter(species)
    ref_sum = sum(c * ISOLATED_ATOM_ENERGIES.get(el, 0) for el, c in counts.items())

    # Check current atomization energy
    current_atm = (float(np.mean(valid)) - ref_sum) / n
    if -20.0 < current_atm < 0.0:
        return 0.0   # already compatible

    # Compute shift so mean atm energy = target
    shift = (target_atm_per_atom * n + ref_sum) - float(np.mean(valid))
    return float(shift)


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
        coords: np.ndarray,                      # (window_size, n_atoms, 3)
        energies: Optional[np.ndarray] = None,   # (window_size,)
        species: Optional[List[str]] = None,     # length n_atoms, e.g. ['Sb','Sb',...,'Te','Cr']
    ) -> Dict[str, float]:
        """Extract all features from one window. Returns flat dict."""
        feats = {}
        feats.update(self._displacement_stats(coords))
        feats.update(self._dynamics_features(coords))
        feats.update(self._frequency_features(coords))
        feats.update(self._msd_features(coords))
        feats.update(self._structural_integrity_features(coords))
        feats.update(self._vacf_features(coords))
        if energies is not None and not np.all(np.isnan(energies)):
            feats.update(self._energy_features(energies, species))
        else:
            feats.update({
                'energy_mean': np.nan,
                'energy_std': np.nan,
                'energy_trend': np.nan,
            })
        return feats

    def extract_all_windows(
        self,
        coords: np.ndarray,                      # (n_frames, n_atoms, 3)
        energies: Optional[np.ndarray] = None,
        species: Optional[List[str]] = None,     # length n_atoms; constant across frames
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
            feats = self.extract_window(w_coords, w_energies, species)
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

    def _energy_features(
        self,
        energies: np.ndarray,
        species: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Per-atom atomization energy features.

        Converts raw DFT total energies (eV) to per-atom atomization energies:
            E_atm/atom = (E_tot - Σ_i N_i * E0_i) / N_total

        where E0_i are the isolated-atom DFT reference energies from
        ISOLATED_ATOM_ENERGIES.  This removes the linear scaling with system
        size so that trajectories with different atom counts are directly
        comparable, and the energy mean/std/trend become physically meaningful
        bonding-energy quantities (~-3 to -5 eV/atom for Sb₂Te₃).

        If species is None or contains unknown elements, falls back to
        raw-energy-per-atom (E_tot / N_atoms) as a degraded but size-normalised
        alternative.
        """
        valid_mask = ~np.isnan(energies)
        valid = energies[valid_mask]
        if len(valid) < 2:
            return {'energy_mean': np.nan, 'energy_std': np.nan, 'energy_trend': np.nan}

        # --- build per-atom reference energy offset ---
        ref_energy_per_frame = 0.0
        n_atoms = len(species) if species is not None else 0

        if species and n_atoms > 0:
            # Count each element
            counts: Dict[str, int] = {}
            for sp in species:
                counts[sp] = counts.get(sp, 0) + 1

            # Sum reference energies for known elements
            ref_sum = sum(
                cnt * ISOLATED_ATOM_ENERGIES[el]
                for el, cnt in counts.items()
                if el in ISOLATED_ATOM_ENERGIES
            )
            # Check if all elements are covered
            unknown = [el for el in counts if el not in ISOLATED_ATOM_ENERGIES]
            if unknown:
                # Partial fallback: use raw E/atom
                ref_sum = 0.0
                n_atoms = 0  # triggers raw /atom below

            ref_energy_per_frame = ref_sum
        else:
            n_atoms = 0  # no species info

        if n_atoms > 0:
            # True per-atom atomization energy (eV/atom)
            norm_energies = (valid - ref_energy_per_frame) / n_atoms
        else:
            # Degraded fallback: just divide by total atoms from coords shape
            # (still size-normalised, but not atomization energy)
            n_atoms_fallback = max(energies.size, 1)
            norm_energies = valid / n_atoms_fallback

        # Physical plausibility check: E_atm/atom for a bound solid must be negative.
        # A positive value indicates the file was calculated with DIFFERENT
        # pseudopotentials than the isolated-atom references (incompatible energy scales),
        # most commonly seen in "bulk" trajectories merged from different DFT setups.
        # In that case, return NaN so the imputer uses the training-set mean rather
        # than a spurious +900 eV/atom value that would corrupt the anomaly scores.
        mean_atm = float(np.mean(norm_energies))
        if mean_atm > 0.0:
            return {'energy_mean': np.nan, 'energy_std': np.nan, 'energy_trend': np.nan}

        trend = float(np.polyfit(np.arange(len(norm_energies)), norm_energies, 1)[0])
        return {
            'energy_mean': float(np.mean(norm_energies)),
            'energy_std': float(np.std(norm_energies)),
            'energy_trend': trend,
        }

    def _structural_integrity_features(self, coords: np.ndarray) -> Dict[str, float]:
        """
        Physics-motivated structural integrity features derived from pairwise distances.

        min_interatomic_dist  — minimum pairwise atomic distance over the window (Å).
            Detects atomic clashes / core-repulsion violations.  Any value below ~1.5 Å
            for heavy atoms signals a nonsensical configuration.

        rdf_first_peak_pos   — position of the first peak in the windowed radial
            distribution function (Å).  Shifts indicate bond-length drift or a
            structural phase change.

        rdf_first_peak_height — normalised height of the first RDF peak.
            Peak collapse or broadening indicates order→disorder transition; large
            values relative to AIMD baseline flag anomalous configurations.
        """
        n_frames, n_atoms, _ = coords.shape

        # For very large supercells (n_atoms > 256), computing all O(n²) pairwise
        # distances requires hundreds of MB per window and will OOM.  Subsample
        # up to MAX_ATOMS atoms uniformly; distances within the subsample are
        # representative for RDF / min-dist detection purposes.
        MAX_ATOMS = 256
        if n_atoms > MAX_ATOMS:
            rng_idx = np.linspace(0, n_atoms - 1, MAX_ATOMS, dtype=int)
            coords_sub = coords[:, rng_idx, :]
            n_atoms_eff = MAX_ATOMS
        else:
            coords_sub = coords
            n_atoms_eff = n_atoms

        # --- all-pairs distance matrix via ||a-b||² = ||a||² + ||b||² - 2 a·b ---
        # Uses batched BLAS matmul: avoids the (T, N, N, 3) broadcast diff
        # which allocates 3× more memory and is slower for large windows.
        norms_sq = np.einsum('tij,tij->ti', coords_sub, coords_sub)           # (T, M)
        dots     = np.matmul(coords_sub, coords_sub.swapaxes(-1, -2))         # (T, M, M)
        dist_sq  = norms_sq[:, :, None] + norms_sq[:, None, :] - 2.0 * dots  # (T, M, M)
        np.clip(dist_sq, 0.0, None, out=dist_sq)      # numerical safety (no sqrt of negatives)
        # Zero the diagonal so it doesn't pollute min search
        diag_idx = np.arange(n_atoms_eff)
        dist_sq[:, diag_idx, diag_idx] = np.inf
        dist_mat = np.sqrt(dist_sq)                                            # (T, M, M)

        min_dist = float(np.min(dist_mat))

        # --- windowed RDF: subsample up to 10 frames to reduce histogram cost ---
        rdf_n    = min(n_frames, 10)
        rdf_idx  = np.linspace(0, n_frames - 1, rdf_n, dtype=int)
        triu     = np.triu_indices(n_atoms_eff, k=1)
        all_dists = dist_mat[rdf_idx][:, triu[0], triu[1]].ravel()   # (rdf_n * n_pairs,)

        r_min, r_max, n_bins = 1.5, 8.0, 100
        valid = all_dists[(all_dists >= r_min) & (all_dists < r_max)]

        if len(valid) < 10:
            return {
                'min_interatomic_dist': min_dist,
                'rdf_first_peak_pos': np.nan,
                'rdf_first_peak_height': np.nan,
            }

        counts, edges = np.histogram(valid, bins=n_bins, range=(r_min, r_max))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0])

        # Normalise by shell volume ∝ r² to obtain a proper g(r) shape
        rdf = counts.astype(float) / (bin_centers ** 2 + 1e-10) / (bin_width + 1e-10)
        rdf_mean = rdf.mean()
        rdf = rdf / (rdf_mean + 1e-10)               # normalise so bulk → 1

        # First peak: highest bin in the first 40 % of the r range
        search_end = max(1, int(n_bins * 0.40))
        peak_idx = int(np.argmax(rdf[:search_end]))

        return {
            'min_interatomic_dist': min_dist,
            'rdf_first_peak_pos': float(bin_centers[peak_idx]),
            'rdf_first_peak_height': float(rdf[peak_idx]),
        }

    def _vacf_features(self, coords: np.ndarray) -> Dict[str, float]:
        """
        Velocity Auto-Correlation Function (VACF) features.
        Velocities are approximated as frame-to-frame coordinate differences.

        vacf_initial_decay   — normalised VACF drop from τ=0 to τ=1 (i.e.
            VACF[0] − VACF[1]) / VACF[0].  Large values indicate stiff, high-
            frequency vibrations or erratic atomic motion characteristic of MLFF
            instability.

        vacf_zero_crossing   — first zero-crossing time expressed as a fraction
            of the maximum lag searched.  Short crossing time ↔ high-frequency
            oscillations; absence of crossing (value=1.0) ↔ strongly correlated
            drift (catastrophic displacement).
        """
        vel = np.diff(coords, axis=0)   # (T-1, n_atoms, 3)
        T = vel.shape[0]

        if T < 4:
            return {'vacf_initial_decay': np.nan, 'vacf_zero_crossing': np.nan}

        max_lag = min(T // 2, 20)

        # VACF[τ] = mean over time origins and atoms of v(t)·v(t+τ)
        vacf = np.empty(max_lag)
        for lag in range(max_lag):
            # dot product over 3-D velocity components: (T-lag, n_atoms)
            dot = np.einsum('taj,taj->ta', vel[:T - lag], vel[lag:T])
            vacf[lag] = float(np.mean(dot))

        norm = vacf[0] if abs(vacf[0]) > 1e-30 else 1.0
        vacf_norm = vacf / norm

        # Initial decay fraction
        initial_decay = float(vacf_norm[0] - vacf_norm[1]) if max_lag > 1 else 0.0

        # First zero crossing (sign change between consecutive lags)
        zero_crossing = 1.0  # default: no crossing found
        for i in range(1, max_lag):
            if vacf_norm[i - 1] * vacf_norm[i] <= 0:
                zero_crossing = float(i) / max_lag
                break

        return {
            'vacf_initial_decay': initial_decay,
            'vacf_zero_crossing': zero_crossing,
        }
