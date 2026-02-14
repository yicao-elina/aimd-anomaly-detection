"""
Data quality assessment module for trajectory data.
Checks for anomalies, missing values, and physical validity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.loaders import TrajectoryLoader, load_all_trajectories


class DataQualityChecker:
    """Comprehensive quality assessment for trajectory data."""
    
    def __init__(self, min_distance_threshold: float = 1.5):
        """
        Initialize QA checker.
        
        Args:
            min_distance_threshold: Minimum allowed interatomic distance in Angstroms
        """
        self.min_distance_threshold = min_distance_threshold
        self.results = []
        self.all_data = None
    
    def load_and_assess_all(self, data_dirs: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Load all trajectories and run comprehensive assessment.
        Returns both detailed results and summary stats.
        """
        print("=" * 80)
        print("STEP 1: DATA INVENTORY & QUALITY ASSESSMENT")
        print("=" * 80)
        print()
        
        # Load all trajectories
        print("📂 Loading trajectories from:")
        for d in data_dirs:
            print(f"   • {d}")
        print()
        
        self.all_data = load_all_trajectories(data_dirs)
        print()
        
        # Assess each trajectory
        print("🔍 Running quality checks:")
        print("-" * 80)
        
        summary_stats = {
            'total_files': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'total_frames': 0,
            'total_atoms': 0,
            'quality_issues': []
        }
        
        for idx, row in self.all_data.iterrows():
            if 'error' in row and pd.notna(row['error']):
                summary_stats['failed_loads'] += 1
                summary_stats['quality_issues'].append(f"Failed to load: {row['filename']}")
                continue
            
            summary_stats['successful_loads'] += 1
            summary_stats['total_files'] += 1
            summary_stats['total_frames'] += row['n_frames']
            summary_stats['total_atoms'] += row['n_atoms']
            
            # Run checks
            qa_result = self._check_trajectory(
                row['filename'],
                row['coordinates'],
                row['energies'],
                row['temperature_K'],
                row['configuration']
            )
            
            self.results.append(qa_result)
        
        print()
        print("-" * 80)
        
        return self.all_data, summary_stats, self.results
    
    def _check_trajectory(self, filename: str, coordinates: np.ndarray, 
                          energies: np.ndarray, temp_K: int = None, 
                          config: str = None) -> Dict:
        """Run all QA checks on a single trajectory."""
        
        n_frames, n_atoms, _ = coordinates.shape
        
        checks = {
            'filename': filename,
            'temperature_K': temp_K,
            'configuration': config,
            'n_frames': n_frames,
            'n_atoms': n_atoms,
            'file_size_bytes': coordinates.nbytes + energies.nbytes,
            'checks': {}
        }
        
        # Check 1: Completeness
        has_nan_coords = np.any(np.isnan(coordinates))
        has_nan_energies = np.any(np.isnan(energies))
        checks['checks']['completeness'] = {
            'passed': not (has_nan_coords or has_nan_energies),
            'nan_coordinates': int(np.isnan(coordinates).sum()),
            'nan_energies': int(np.isnan(energies).sum()),
            'message': 'No missing values' if not (has_nan_coords or has_nan_energies) 
                      else f"Found NaN values: {int(np.isnan(coordinates).sum())} in coords, {int(np.isnan(energies).sum())} in energies"
        }
        
        # Check 2: Minimum distances (physical validity)
        min_dist, min_dist_frame = self._check_minimum_distances(coordinates)
        min_dist_ok = min_dist >= self.min_distance_threshold
        checks['checks']['minimum_distance'] = {
            'passed': min_dist_ok,
            'min_distance_angstrom': float(min_dist),
            'worst_frame_idx': int(min_dist_frame),
            'threshold_angstrom': self.min_distance_threshold,
            'message': f'Min distance {min_dist:.3f} Å (Frame {min_dist_frame})' + 
                      ('✓' if min_dist_ok else ' ⚠️ BELOW THRESHOLD')
        }
        
        # Check 3: Energy conservation
        if not has_nan_energies:
            energy_drift = self._check_energy_drift(energies)
            checks['checks']['energy_drift'] = {
                'passed': energy_drift < 5.0,  # Allow 5% drift
                'max_drift_percent': float(energy_drift),
                'min_energy': float(np.min(energies)),
                'max_energy': float(np.max(energies)),
                'message': f'Energy drift: {energy_drift:.2f}%'
            }
        else:
            checks['checks']['energy_drift'] = {
                'passed': False,
                'message': 'Cannot assess: NaN energies'
            }
        
        # Check 4: Coordinate stability
        coord_std = np.std(coordinates, axis=0).mean()
        checks['checks']['coordinate_stability'] = {
            'passed': coord_std > 0.01,
            'avg_std_angstrom': float(coord_std),
            'message': f'Atomic motion (std): {coord_std:.4f} Å'
        }
        
        # Overall status
        all_passed = all(c.get('passed', False) for c in checks['checks'].values())
        checks['overall_status'] = 'PASS' if all_passed else 'WARN'
        
        # Print summary
        status_symbol = '✓' if all_passed else '⚠️'
        print(f"{status_symbol} {filename:50s} | {n_frames:5d} frames | {n_atoms:3d} atoms | {checks['overall_status']}")
        
        return checks
    
    @staticmethod
    def _check_minimum_distances(coordinates: np.ndarray) -> Tuple[float, int]:
        """
        Find minimum interatomic distance across all frames.
        Returns (min_distance, frame_index_with_min_distance)
        """
        min_dist = np.inf
        min_frame = 0
        
        for frame_idx, frame in enumerate(coordinates):
            # Compute pairwise distances
            diff = frame[:, np.newaxis, :] - frame[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            
            # Get minimum non-zero distance (exclude self-distances on diagonal)
            np.fill_diagonal(distances, np.inf)
            frame_min = np.min(distances)
            
            if frame_min < min_dist:
                min_dist = frame_min
                min_frame = frame_idx
        
        return float(min_dist), int(min_frame)
    
    @staticmethod
    def _check_energy_drift(energies: np.ndarray) -> float:
        """
        Check energy drift as percentage of energy range.
        Lower is better (< 5% is good).
        """
        energy_range = np.max(energies) - np.min(energies)
        if energy_range == 0:
            return 0.0
        
        # Fit linear trend to energies
        x = np.arange(len(energies))
        coeffs = np.polyfit(x, energies, 1)
        slope = coeffs[0]
        
        drift_percent = abs(slope) * len(energies) / energy_range * 100
        return float(drift_percent)
    
    def generate_markdown_report(self, output_path: str) -> None:
        """Generate human-readable quality report in markdown."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            f.write("# Data Quality Assessment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            total_frames = sum(r['n_frames'] for r in self.results)
            total_atoms = sum(r['n_atoms'] for r in self.results)
            total_files = len(self.results)
            
            f.write(f"- **Total Files Loaded**: {total_files}\n")
            f.write(f"- **Total Frames**: {total_frames:,}\n")
            f.write(f"- **Total Atoms (across all)**: {total_atoms:,}\n")
            f.write(f"- **Assessment Timestamp**: {datetime.now().isoformat()}\n\n")
            
            # Quality summary
            passed = sum(1 for r in self.results if r['overall_status'] == 'PASS')
            warned = sum(1 for r in self.results if r['overall_status'] == 'WARN')
            
            f.write("## Quality Summary\n\n")
            f.write(f"| Status | Count | Percentage |\n")
            f.write(f"|--------|-------|------------|\n")
            f.write(f"| ✓ PASS | {passed} | {100*passed/total_files:.0f}% |\n")
            f.write(f"| ⚠️ WARN | {warned} | {100*warned/total_files:.0f}% |\n")
            f.write(f"| **Total** | **{total_files}** | **100%** |\n\n")
            
            # Detailed results
            f.write("## Detailed Assessment by File\n\n")
            
            for result in self.results:
                f.write(f"### {result['filename']}\n\n")
                f.write(f"- **Temperature**: {result['temperature_K']} K\n")
                f.write(f"- **Configuration**: {result['configuration']}\n")
                f.write(f"- **Frames**: {result['n_frames']}\n")
                f.write(f"- **Atoms per Frame**: {result['n_atoms']}\n")
                f.write(f"- **File Size**: {result['file_size_bytes'] / 1e6:.1f} MB\n")
                f.write(f"- **Status**: {result['overall_status']}\n\n")
                
                f.write("#### Quality Checks\n\n")
                for check_name, check_result in result['checks'].items():
                    status = "✓ PASS" if check_result.get('passed', False) else "⚠️ WARN"
                    message = check_result.get('message', 'N/A')
                    f.write(f"- **{check_name.replace('_', ' ').title()}**: {status}\n")
                    f.write(f"  - {message}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if warned > 0:
                f.write("- Some trajectories showed warnings:\n")
                for result in self.results:
                    if result['overall_status'] == 'WARN':
                        f.write(f"  - {result['filename']}\n")
                        for check_name, check in result['checks'].items():
                            if not check.get('passed', True):
                                f.write(f"    - {check_name}: {check.get('message', 'Issue detected')}\n")
                f.write("- Review these files for potential anomalies or data quality issues.\n")
            else:
                f.write("- All trajectories pass quality checks. ✓\n")
                f.write("- Data is ready for feature extraction and anomaly detection.\n")
            
            f.write("\n---\n")
            f.write("✓ Report generated by AIMD Anomaly Detection Framework\n")


def create_inventory_csv(all_data: pd.DataFrame, results: List[Dict], output_path: str) -> None:
    """Create CSV inventory of all loaded data."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    inventory = []
    result_by_filename = {r['filename']: r for r in results}
    
    for idx, row in all_data.iterrows():
        if pd.isna(row['n_frames']):
            # Failed load
            inventory.append({
                'filename': row['filename'],
                'filepath': row['filepath'],
                'type': row['type'] if pd.notna(row['type']) else 'unknown',
                'temperature_K': row['temperature_K'] if pd.notna(row['temperature_K']) else None,
                'configuration': row['configuration'] if pd.notna(row['configuration']) else None,
                'n_frames': None,
                'n_atoms': None,
                'total_particles': None,
                'file_size_MB': None,
                'quality_status': 'FAILED',
                'error': row['error'] if 'error' in row else 'Unknown error'
            })
        else:
            # Successful load
            result = result_by_filename.get(row['filename'], {})
            overall_status = result.get('overall_status', 'UNKNOWN')
            
            inventory.append({
                'filename': row['filename'],
                'filepath': row['filepath'],
                'type': row['type'],
                'temperature_K': row['temperature_K'],
                'configuration': row['configuration'],
                'n_frames': int(row['n_frames']),
                'n_atoms': int(row['n_atoms']),
                'total_particles': int(row['n_frames'] * row['n_atoms']),
                'file_size_MB': row['coordinates'].nbytes / 1e6 if 'coordinates' in row else None,
                'quality_status': overall_status,
                'error': None
            })
    
    df = pd.DataFrame(inventory)
    df.to_csv(output, index=False)
    print(f"✓ Inventory saved to {output}")
    
    return df


if __name__ == '__main__':
    # Configuration
    data_dirs = [
        'data/raw/temperature',
        'data/raw/concentration',
        'data/raw/mlff'
    ]
    
    output_csv = 'data/processed/data_inventory.csv'
    output_report = 'results/reports/data_quality_report.md'
    
    # Run assessment
    checker = DataQualityChecker(min_distance_threshold=1.5)
    all_data, summary_stats, qa_results = checker.load_and_assess_all(data_dirs)
    
    # Generate outputs
    print("\n📊 Generating reports...")
    inventory_df = create_inventory_csv(all_data, qa_results, output_csv)
    checker.generate_markdown_report(output_report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files examined: {summary_stats['total_files']}")
    print(f"Successful loads: {summary_stats['successful_loads']}")
    print(f"Failed loads: {summary_stats['failed_loads']}")
    print(f"Total frames across all files: {summary_stats['total_frames']:,}")
    print(f"Total atoms (unique per file): {len(qa_results)} files")
    
    passed = sum(1 for r in qa_results if r['overall_status'] == 'PASS')
    warned = sum(1 for r in qa_results if r['overall_status'] == 'WARN')
    print(f"\nQuality Status:")
    print(f"  ✓ PASS: {passed}/{len(qa_results)}")
    print(f"  ⚠️ WARN: {warned}/{len(qa_results)}")
    
    print(f"\n📁 Outputs:")
    print(f"  • {output_csv}")
    print(f"  • {output_report}")
    print("\n" + "=" * 80)
