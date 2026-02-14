"""
Data loaders for AIMD trajectories and other time-series data.
Abstract interface allows extension to GPU metrics, performance data, etc.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Any
import re


class DataLoader(ABC):
    """Base class for generic time-series data loaders."""
    
    @abstractmethod
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load data from file and return standardized dict."""
        pass
    
    @abstractmethod
    def get_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from filename or file content."""
        pass


class TrajectoryLoader(DataLoader):
    """Loader for AIMD trajectory files (.xyz format)."""
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Load XYZ trajectory file.
        
        Returns dict with:
            - 'coordinates': np.array of shape (n_frames, n_atoms, 3)
            - 'species': list of atomic species (one per atom)
            - 'energies': np.array of shape (n_frames,) if available
            - 'n_frames': int
            - 'n_atoms': int
            - 'metadata': dict with file-level metadata
        """
        filepath = Path(filepath)
        
        frames = []
        energies = []
        species = None
        n_atoms = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        while idx < len(lines):
            # Parse frame header
            try:
                n_atoms = int(lines[idx].strip())
            except (ValueError, IndexError):
                break
            idx += 1
            
            if idx >= len(lines):
                break
            
            # Parse comment line (contains energy and other metadata)
            comment = lines[idx]
            energy = self._extract_energy(comment)
            energies.append(energy)
            idx += 1
            
            # Parse atomic coordinates
            frame_coords = []
            frame_species = []
            
            for atom_idx in range(n_atoms):
                if idx >= len(lines):
                    raise ValueError(f"Unexpected end of file at frame, atom {atom_idx}/{n_atoms}")
                
                parts = lines[idx].split()
                if len(parts) < 4:
                    break
                
                species_name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                frame_coords.append([x, y, z])
                
                if atom_idx == 0:
                    # Initialize species list on first frame
                    if species is None:
                        species = []
                
                if len(frame_species) < len(species) or len(frame_species) < atom_idx + 1:
                    frame_species.append(species_name)
                
                idx += 1
            
            if len(frame_coords) == n_atoms:
                frames.append(frame_coords)
                if species is None or len(species) == 0:
                    species = frame_species
        
        if not frames:
            raise ValueError(f"No frames found in {filepath}")
        
        coordinates = np.array(frames, dtype=np.float32)  # (n_frames, n_atoms, 3)
        energies = np.array(energies, dtype=np.float32)
        
        return {
            'coordinates': coordinates,
            'species': species,
            'energies': energies,
            'n_frames': coordinates.shape[0],
            'n_atoms': coordinates.shape[1],
            'metadata': self.get_metadata(str(filepath))
        }
    
    def get_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract temperature and configuration from filename."""
        filepath = Path(filepath)
        filename = filepath.stem
        
        metadata = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'temperature_K': None,
            'configuration': None,
            'type': None,  # 'temperature' or 'concentration'
        }
        
        # Extract temperature if present
        temp_match = re.search(r'(\d+)K', filename)
        if temp_match:
            metadata['temperature_K'] = int(temp_match.group(1))
            metadata['type'] = 'temperature'
        
        # Extract configuration name
        if '2L_o3_t' in filename:
            match = re.search(r'2L_o3_t\d+', filename)
            metadata['configuration'] = match.group(0)
            metadata['type'] = 'concentration'
        elif '2L_octo_Cr' in filename:
            match = re.search(r'2L_octo_Cr\d+', filename)
            metadata['configuration'] = match.group(0)
            metadata['type'] = 'concentration'
        
        if 'mlff' in filename.lower():
            metadata['type'] = 'mlff'
            metadata['configuration'] = 'MLFF-NEB'
        
        return metadata
    
    @staticmethod
    def _extract_energy(comment_line: str) -> float:
        """Extract energy value from comment line."""
        match = re.search(r'energy=([+-]?\d+\.?\d*)', comment_line)
        if match:
            return float(match.group(1))
        return np.nan


class DistanceMatrixLoader(DataLoader):
    """Loader for pre-computed distance matrices (.csv format)."""
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load distance matrix CSV file."""
        filepath = Path(filepath)
        
        df = pd.read_csv(filepath, index_col=0)
        n_frames = len(df)
        
        return {
            'distances': df.values,
            'n_frames': n_frames,
            'metadata': self.get_metadata(str(filepath))
        }
    
    def get_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        filepath = Path(filepath)
        filename = filepath.stem
        
        return {
            'filename': filepath.name,
            'filepath': str(filepath),
            'type': 'distance_matrix'
        }


def load_all_trajectories(data_dirs: list) -> pd.DataFrame:
    """
    Load all trajectory files from specified directories.
    Returns a DataFrame with file info and loaded coordinates.
    """
    loader = TrajectoryLoader()
    results = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️  Directory not found: {data_path}")
            continue
        
        for xyz_file in sorted(data_path.glob('*.xyz')):
            try:
                print(f"Loading: {xyz_file.name}...", end=" ")
                trajectory = loader.load(str(xyz_file))
                
                results.append({
                    'filename': trajectory['metadata']['filename'],
                    'filepath': str(xyz_file),
                    'n_frames': trajectory['n_frames'],
                    'n_atoms': trajectory['n_atoms'],
                    'temperature_K': trajectory['metadata']['temperature_K'],
                    'configuration': trajectory['metadata']['configuration'],
                    'type': trajectory['metadata']['type'],
                    'coordinates': trajectory['coordinates'],
                    'energies': trajectory['energies'],
                    'species': trajectory['species']
                })
                print(f"✓ ({trajectory['n_frames']} frames)")
            
            except Exception as e:
                print(f"✗ Error: {str(e)[:100]}")
                results.append({
                    'filename': xyz_file.name,
                    'filepath': str(xyz_file),
                    'error': str(e),
                    'n_frames': None,
                    'n_atoms': None,
                    'temperature_K': None,
                    'configuration': None,
                    'type': None
                })
    
    return pd.DataFrame(results)
