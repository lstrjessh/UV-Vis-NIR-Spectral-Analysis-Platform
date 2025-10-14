"""
Shared utilities for Mathematical Modelling and Predict Concentration applications.
"""
import re
from typing import Optional

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
    patterns = [
        r'(\d+\.?\d*)\s*(?:mL|ml|ppm|ppb|mg/L|mg_L|μg|ug|mM|μM|uM|nM|M|g/L|g_L|mg|g|L)',  # number with units
        r'(\d+\.?\d*)_(?:mL|ml|ppm|ppb|mg/L|mg_L|μg|ug|mM|μM|uM|nM|M|g/L|g_L|mg|g|L)',   # number_unit
        r'(\d+\.?\d*)-(?:mL|ml|ppm|ppb|mg|μg|ug|mM|μM|uM|nM|M|g|L)',   # number-unit
        r'[A-Za-z]{1,2}[_\-\s]*(\d+\.?\d*)[_\-\s]*(?:ppm|ppb|mg|μg|ug|mM|μM|uM|nM|M|g|L)',  # Element_number_unit (e.g., Cu_0.1ppm)
        r'concentration[_\-\s]*(\d+\.?\d*)',  # concentration_number or concentration-number
        r'conc[_\-\s]*(\d+\.?\d*)',  # conc_number or conc-number
        r'sample[_\-\s]*(\d+\.?\d*)',  # sample_number
        r'_(\d+\.?\d*)_',  # number between underscores
        r'^(\d+\.?\d*)_',  # number at start followed by underscore
        r'_(\d+\.?\d*)$',  # number at end preceded by underscore
        r'^(\d+\.?\d*)$',  # just a number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None
