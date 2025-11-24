"""
Dataset loading with instrument labels.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import re


def load_nsynth_with_labels(cqt_path, n_samples=None):
    """
    Load NSynth CQT data with instrument family labels.
    
    NSynth filenames follow pattern:
    bass_electronic_018-022-050.npy
    ^--- instrument family
    
    Args:
        cqt_path: Path to directory containing .npy CQT files
        n_samples: Optional, limit number of samples loaded
        
    Returns:
        cqt_data: [n_samples, n_bins] numpy array
        labels: [n_samples] list of instrument families
        metadata: Dictionary with additional info
    """
    cqt_path = Path(cqt_path)
    
    if not cqt_path.exists():
        raise FileNotFoundError(f"CQT path not found: {cqt_path}")
    
    # Find all .npy files (excluding metadata files)
    npy_files = sorted([
        f for f in cqt_path.glob("*.npy") 
        if not f.stem.endswith('_metadata')
    ])
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {cqt_path}")
    
    print(f"Found {len(npy_files)} CQT files")
    
    # Limit samples if requested
    if n_samples is not None and n_samples < len(npy_files):
        # Use random sampling for diversity
        import random
        random.seed(42)  # Reproducible
        npy_files = random.sample(npy_files, n_samples)
        print(f"Randomly sampled {n_samples} files")
    
    # Extract labels and load data
    cqt_data_list = []
    labels = []
    filenames = []
    label_counts = defaultdict(int)
    
    for npy_file in npy_files:
        # Extract instrument family from filename
        # Pattern: instrument_type_numbers.npy
        filename = npy_file.stem
        
        # Extract first word as instrument family
        match = re.match(r'^([a-zA-Z]+)_', filename)
        if match:
            instrument_family = match.group(1)
        else:
            # Fallback: use first part before underscore
            parts = filename.split('_')
            instrument_family = parts[0] if parts else 'unknown'
        
        # Load CQT data
        try:
            cqt = np.load(npy_file)
            
            # Handle different possible shapes
            if cqt.ndim == 2:
                # If multiple frames, take mean across time
                cqt_mean = np.mean(cqt, axis=0)
            elif cqt.ndim == 1:
                cqt_mean = cqt
            else:
                print(f"Unexpected shape for {filename}: {cqt.shape}, skipping")
                continue
            
            cqt_data_list.append(cqt_mean)
            labels.append(instrument_family)
            filenames.append(filename)
            label_counts[instrument_family] += 1
            
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue
    
    if len(cqt_data_list) == 0:
        raise ValueError("No valid CQT data loaded")
    
    # Convert to numpy arrays
    cqt_data = np.array(cqt_data_list, dtype=np.float32)
    labels = np.array(labels)
    
    # Create metadata
    unique_labels = np.unique(labels)
    metadata = {
        'n_samples': len(cqt_data),
        'n_bins': cqt_data.shape[1],
        'instrument_families': unique_labels.tolist(),
        'n_families': len(unique_labels),
        'label_distribution': dict(label_counts),
        'filenames': filenames,
        'cqt_path': str(cqt_path)
    }
    
    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {metadata['n_samples']}")
    print(f"CQT bins: {metadata['n_bins']}")
    print(f"Instrument families: {metadata['n_families']}")
    print("\nLabel distribution:")
    for family in sorted(unique_labels):
        count = label_counts[family]
        percentage = 100 * count / len(labels)
        print(f"  {family:15s}: {count:4d} ({percentage:5.1f}%)")
    
    return cqt_data, labels, metadata


def stratified_sample(cqt_data, labels, n_per_class):
    """
    Sample equal number from each instrument class for balanced visualization.
    """
    unique_labels = np.unique(labels)
    
    sampled_data = []
    sampled_labels = []
    
    for label in unique_labels:
        # Get indices for this class
        indices = np.where(labels == label)[0]
        
        # Sample n_per_class (or all if fewer available)
        n_available = len(indices)
        n_to_sample = min(n_per_class, n_available)
        
        if n_to_sample < n_per_class:
            print(f"Warning: Only {n_available} samples available for {label}, requested {n_per_class}")
        
        # Random sampling without replacement
        sampled_indices = np.random.choice(indices, size=n_to_sample, replace=False)
        
        # Add to lists
        sampled_data.append(cqt_data[sampled_indices])
        sampled_labels.extend([label] * n_to_sample)
    
    # Concatenate all samples
    sampled_data = np.vstack(sampled_data)
    sampled_labels = np.array(sampled_labels)
    
    print(f"\nStratified sampling:")
    print(f"  Requested per class: {n_per_class}")
    print(f"  Total sampled: {len(sampled_data)}")
    print(f"  Classes: {len(unique_labels)}")
    
    return sampled_data, sampled_labels


def get_label_colors(labels):
    """
    Generate consistent colors for instrument families.
    """
    import matplotlib.pyplot as plt
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Use a colormap with good distinction
    cmap = plt.cm.get_cmap('tab20' if n_labels <= 20 else 'hsv')
    
    # Create color map
    color_map = {}
    for i, label in enumerate(unique_labels):
        color_map[label] = cmap(i / max(n_labels - 1, 1))
    
    # Map labels to colors
    colors = np.array([color_map[label] for label in labels])
    
    return colors, color_map


def load_single_cqt_file(cqt_file_path):
    """
    Load a single CQT file and extract its instrument label.
    """
    cqt_file_path = Path(cqt_file_path)
    
    # Extract label from filename
    filename = cqt_file_path.stem
    match = re.match(r'^([a-zA-Z]+)_', filename)
    label = match.group(1) if match else 'unknown'
    
    # Load data
    cqt_data = np.load(cqt_file_path)
    
    return cqt_data, label, filename


def filter_by_instrument(cqt_data, labels, instrument_family):
    """
    Filter dataset to only include specific instrument family.
    
    Args:
        cqt_data: [n_samples, n_bins] array
        labels: [n_samples] array of labels
        instrument_family: String, e.g., 'bass', 'guitar', 'piano'
        
    Returns:
        filtered_data: Subset of cqt_data
        filtered_labels: Subset of labels
    """
    mask = labels == instrument_family
    filtered_data = cqt_data[mask]
    filtered_labels = labels[mask]
    
    print(f"Filtered to {instrument_family}: {len(filtered_data)} samples")
    
    return filtered_data, filtered_labels