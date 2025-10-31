from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import io
import numpy as np
import pandas as pd


EPSILON = 1e-10


@dataclass
class Spectrum:
    wavelengths: np.ndarray
    counts: np.ndarray


def read_spectral_file_from_path(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, 'rb') as f:
            content = f.read()
        return read_spectral_file_from_bytes(content)
    except Exception:
        return None


def read_spectral_file_from_bytes(content: bytes) -> Optional[pd.DataFrame]:
    try:
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')

        buf = io.StringIO(text)
        df: Optional[pd.DataFrame] = None
        for delimiter in [None, '\t', ' ', ',', ';']:
            buf.seek(0)
            try:
                if delimiter is None:
                    df = pd.read_csv(buf, delim_whitespace=True)
                else:
                    df = pd.read_csv(buf, delimiter=delimiter)
                if len(df.columns) >= 2:
                    df.columns = ['Nanometers', 'Counts'] + list(df.columns[2:])
                    break
            except Exception:
                continue
        if df is None or df.empty:
            return None
        # Clean
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df = df.dropna(subset=df.columns[:2])
        if df.empty:
            return None
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        df.columns = ['Nanometers', 'Counts'] + list(df.columns[2:])
        return df
    except Exception:
        return None


def average_counts_dataframes(dataframes: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not dataframes:
        return None
    if len(dataframes) == 1:
        return dataframes[0].copy()
    ref = dataframes[0]
    ref_wl = ref['Nanometers'].values
    for df in dataframes[1:]:
        if not np.allclose(df['Nanometers'].values, ref_wl, rtol=1e-3):
            return None
    result = ref.copy()
    counts_matrix = np.column_stack([df['Counts'].values for df in dataframes])
    result['Counts'] = np.mean(counts_matrix, axis=1)
    return result


def calculate_absorbance(ref_df: pd.DataFrame, sample_df: pd.DataFrame, dark_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    try:
        if ref_df is None or sample_df is None:
            return None
        ref_wl = ref_df['Nanometers'].values
        sample_wl = sample_df['Nanometers'].values
        if not np.allclose(ref_wl, sample_wl, rtol=1e-3):
            return None
        result = pd.DataFrame({
            'Nanometers': ref_wl,
            'Reference_Counts': ref_df['Counts'].values,
            'Sample_Counts': sample_df['Counts'].values,
        })
        dark_counts = np.zeros_like(ref_wl, dtype=float)
        if dark_df is not None:
            if not np.allclose(dark_df['Nanometers'].values, ref_wl, rtol=1e-3):
                return None
            dark_counts = dark_df['Counts'].values.astype(float)
            result['Dark_Counts'] = dark_counts
        ref_corr = np.clip(result['Reference_Counts'] - dark_counts, EPSILON, None)
        samp_corr = np.clip(result['Sample_Counts'] - dark_counts, EPSILON, None)
        result['Reference_Corrected'] = ref_corr
        result['Sample_Corrected'] = samp_corr
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.log10(ref_corr / samp_corr)
            A = np.nan_to_num(A, nan=0.0, posinf=5.0, neginf=0.0)
        result['Absorbance'] = A
        return result
    except Exception:
        return None


def savgol_smooth(values: np.ndarray, window: int, poly: int) -> Optional[np.ndarray]:
    if window <= 0 or window % 2 == 0:
        return None
    if poly < 0 or poly >= window:
        return None
    if len(values) < window:
        return None
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(values, window, poly)
    except Exception:
        return None


def find_peaks_absorbance(wavelengths: np.ndarray, absorbance: np.ndarray, height: float, distance_nm: float, prominence: float) -> Optional[Tuple[np.ndarray, dict]]:
    if len(wavelengths) < 3 or len(absorbance) < 3:
        return None
    try:
        from scipy.signal import find_peaks
        avg_spacing = float(np.mean(np.diff(wavelengths)))
        distance_pts = max(1, int(distance_nm / avg_spacing)) if avg_spacing > 0 else 1
        peaks, props = find_peaks(
            absorbance,
            height=height if height > 0 else None,
            distance=distance_pts,
            prominence=prominence if prominence > 0 else None,
        )
        return peaks, props
    except Exception:
        return None


