"""
Shared utilities for Mathematical Modelling and Predict Concentration applications.
"""
import re
import numpy as np
from typing import Optional, Tuple

# Common constants
SUPPORTED_EXTENSIONS = ["csv"]
MAX_FILE_SIZE_MB = 50
MAX_POLYNOMIAL_DEGREE = 6

# Color palette for consistent plotting
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]


def extract_concentration_from_filename(filename: str) -> Optional[float]:
    """
    Extract concentration value from filename.
    
    Examples:
    - '0.1mL.csv' -> 0.1
    - '2.5ppm.csv' -> 2.5
    - '10ppb.csv' -> 10
    - '1.5mg_L.csv' -> 1.5
    - '3mg/L.csv' -> 3.0
    - 'sample_0.5_mM.csv' -> 0.5
    - 'concentration_1.2.csv' -> 1.2
    - 'conc-0.8-ppm.csv' -> 0.8
    - 'Cu_0.1ppm.csv' -> 0.1
    - 'Pb_2.5mg_L.csv' -> 2.5
    
    Args:
        filename: Name of the file
        
    Returns:
        Extracted concentration value or None if not found
    """
    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Pattern to match numbers (including decimals) followed by optional units
    # This will match patterns like: 0.1mL, 2.5ppm, 10ppb, 1.5mg_L, 0.5_mM
    # Using improved decimal pattern: \d+(?:\.\d+)? for better matching
    patterns = [
        r'^(\d+(?:\.\d+)?)_[a-zA-Z]',  # number at start followed by underscore and letter (e.g., 1.0_b, 0.9_a)
        r'(\d+(?:\.\d+)?)\s*(?:mL|ml|ppm|ppb|mg/L|mg_L|μg|ug|mM|μM|uM|nM|M|g/L|g_L|mg|g|L)',  # number with units
        r'(\d+(?:\.\d+)?)_(?:mL|ml|ppm|ppb|mg/L|mg_L|μg|ug|mM|μM|uM|nM|M|g/L|g_L|mg|g|L)',   # number_unit
        r'(\d+(?:\.\d+)?)-(?:mL|ml|ppm|ppb|mg|μg|ug|mM|μM|uM|nM|M|g|L)',   # number-unit
        r'[A-Za-z]{1,2}[_\-\s]*(\d+(?:\.\d+)?)[_\-\s]*(?:ppm|ppb|mg|μg|ug|mM|μM|uM|nM|M|g|L)',  # Element_number_unit (e.g., Cu_0.1ppm)
        r'concentration[_\-\s]*(\d+(?:\.\d+)?)',  # concentration_number or concentration-number
        r'conc[_\-\s]*(\d+(?:\.\d+)?)',  # conc_number or conc-number
        r'sample[_\-\s]*(\d+(?:\.\d+)?)',  # sample_number
        r'_(\d+(?:\.\d+)?)_',  # number between underscores
        r'^(\d+(?:\.\d+)?)_',  # number at start followed by underscore
        r'_(\d+(?:\.\d+)?)$',  # number at end preceded by underscore
        r'^(\d+(?:\.\d+)?)$',  # just a number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def kennard_stone_split(X: np.ndarray, y: np.ndarray, train_size: float = 0.8, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Kennard-Stone algorithm for train/test splitting.
    
    The Kennard-Stone algorithm selects training samples that are uniformly distributed
    across the feature space, which is ideal for spectroscopic calibration models.
    
    Algorithm:
    1. Start with two samples that are furthest apart
    2. Iteratively add samples with maximum minimum distance to selected samples
    3. Remaining samples form the test set
    
    Args:
        X: Feature matrix (samples x features)
        y: Target values
        train_size: Proportion of data for training (0.0 to 1.0)
        random_state: Random seed for reproducibility (used for tie-breaking)
        
    Returns:
        X_train, X_test, y_train, y_test
        
    References:
        Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
        Technometrics, 11(1), 137-148.
    """
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_train = int(n_samples * train_size)
    
    # Ensure at least 2 samples in training and 1 in test
    n_train = max(2, min(n_train, n_samples - 1))
    
    # Normalize features for distance calculation
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    # Initialize
    selected_indices = []
    remaining_indices = list(range(n_samples))
    
    # Step 1: Find two samples that are furthest apart
    max_dist = -1
    idx1, idx2 = 0, 1
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(X_normalized[i] - X_normalized[j])
            if dist > max_dist:
                max_dist = dist
                idx1, idx2 = i, j
    
    # Add the two furthest samples
    selected_indices.extend([idx1, idx2])
    remaining_indices.remove(idx1)
    remaining_indices.remove(idx2)
    
    # Step 2: Iteratively select samples with maximum minimum distance
    while len(selected_indices) < n_train and remaining_indices:
        max_min_dist = -1
        best_idx = None
        
        for idx in remaining_indices:
            # Calculate minimum distance to any selected sample
            min_dist = min([
                np.linalg.norm(X_normalized[idx] - X_normalized[sel_idx])
                for sel_idx in selected_indices
            ])
            
            # Track sample with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    # Create train/test splits
    train_indices = np.array(selected_indices)
    test_indices = np.array(remaining_indices)
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
