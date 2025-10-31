"""
Data loading implementations with caching and validation.
"""

import io
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    # Create dummy st object for non-streamlit contexts
    class _DummyStreamlit:
        def warning(self, *args, **kwargs): pass
        def progress(self, *args, **kwargs): return _DummyProgress()
        def expander(self, *args, **kwargs): return _DummyExpander()
        def write(self, *args, **kwargs): pass
        def success(self, *args, **kwargs): pass
    class _DummyProgress:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def progress(self, *args, **kwargs): pass
        def empty(self): pass
    class _DummyExpander:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, *args, **kwargs): pass
    st = _DummyStreamlit()
from ..core.interfaces import IDataLoader
from ..core.data_structures import SpectralData, CalibrationDataset
from ..core.exceptions import FileProcessingError, DataValidationError
import hashlib
from utils.shared_utils import extract_concentration_from_filename


class CSVDataLoader(IDataLoader):
    """CSV file loader for spectral data."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 separator: str = 'auto',
                 decimal: str = '.',
                 cache_enabled: bool = True):
        """
        Initialize CSV loader.
        
        Args:
            encoding: File encoding
            separator: Column separator (auto-detect if 'auto')
            decimal: Decimal separator
            cache_enabled: Enable caching
        """
        self.encoding = encoding
        self.separator = separator
        self.decimal = decimal
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None
    
    def load_file(self, filepath: Union[str, Path]) -> SpectralData:
        """
        Load a single CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            SpectralData object
        """
        filepath = Path(filepath)
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(filepath)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Read file
        df = self._read_csv(filepath)
        
        # Extract data
        wavelengths, absorbance = self._extract_columns(df)
        
        # Extract concentration from filename if present
        concentration = self._extract_concentration(filepath.name)
        
        # Create SpectralData
        data = SpectralData(
            wavelengths=wavelengths,
            absorbance=absorbance,
            filename=filepath.name,
            concentration=concentration,
            metadata={'source_path': str(filepath)}
        )
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = data
        
        return data
    
    def load_multiple(self, filepaths: List[Union[str, Path]]) -> CalibrationDataset:
        """
        Load multiple CSV files.
        
        Args:
            filepaths: List of file paths
            
        Returns:
            CalibrationDataset object
        """
        if not filepaths:
            raise ValueError("No files provided")
        
        spectra = []
        errors = []
        
        for filepath in filepaths:
            try:
                spectrum = self.load_file(filepath)
                spectra.append(spectrum)
            except Exception as e:
                errors.append(f"{Path(filepath).name}: {str(e)}")
        
        if not spectra:
            raise FileProcessingError(f"Failed to load any files. Errors: {errors}")
        
        if errors:
            st.warning(f"Some files failed to load: {errors}")
        
        return CalibrationDataset(
            spectra=spectra,
            name="Loaded Dataset",
            metadata={'n_errors': len(errors), 'errors': errors}
        )
    
    def validate_format(self, filepath: Union[str, Path]) -> bool:
        """
        Validate CSV file format.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            True if valid format
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return False
            
            if filepath.suffix.lower() not in ['.csv', '.txt', '.dat']:
                return False
            
            # Try to read and validate structure
            df = self._read_csv(filepath)
            self._extract_columns(df)
            
            return True
            
        except Exception:
            return False
    
    def _read_csv(self, filepath: Path) -> pd.DataFrame:
        """Read CSV with auto-detection of format."""
        # Try different encodings
        for encoding in [self.encoding, 'latin-1', 'cp1252', 'utf-8']:
            try:
                if self.separator == 'auto':
                    # Auto-detect separator
                    for sep in [',', ';', '\t', ' ']:
                        try:
                            df = pd.read_csv(
                                filepath, 
                                sep=sep, 
                                encoding=encoding,
                                decimal=self.decimal
                            )
                            if len(df.columns) >= 2:
                                return df
                        except:
                            continue
                else:
                    df = pd.read_csv(
                        filepath,
                        sep=self.separator,
                        encoding=encoding,
                        decimal=self.decimal
                    )
                    return df
            except:
                continue
        
        raise FileProcessingError(f"Could not read CSV file: {filepath.name}")
    
    def _extract_columns(self, df: pd.DataFrame) -> tuple:
        """Extract wavelength and absorbance columns."""
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Find wavelength column
        wavelength_col = None
        absorbance_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not wavelength_col and any(kw in col_lower for kw in ['wavelength', 'nanometer', 'nm', 'lambda']):
                wavelength_col = col
            elif not absorbance_col and any(kw in col_lower for kw in ['absorbance', 'abs', 'intensity']):
                absorbance_col = col
        
        if not wavelength_col or not absorbance_col:
            # Try to use first two columns
            if len(df.columns) >= 2:
                wavelength_col = df.columns[0]
                absorbance_col = df.columns[1]
            else:
                raise DataValidationError("Could not identify wavelength and absorbance columns")
        
        # Convert to numeric
        wavelengths = pd.to_numeric(df[wavelength_col], errors='coerce').dropna().values
        absorbance = pd.to_numeric(df[absorbance_col], errors='coerce').dropna().values
        
        if len(wavelengths) != len(absorbance):
            # Align lengths
            min_len = min(len(wavelengths), len(absorbance))
            wavelengths = wavelengths[:min_len]
            absorbance = absorbance[:min_len]
        
        if len(wavelengths) < 2:
            raise DataValidationError("Insufficient valid data points")
        
        return wavelengths, absorbance
    
    def _extract_concentration(self, filename: str) -> Optional[float]:
        """Extract concentration from filename using shared utility function."""
        return extract_concentration_from_filename(filename)
    
    def _get_cache_key(self, filepath: Path) -> str:
        """Generate cache key for file."""
        stat = filepath.stat()
        key_str = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()


class StreamlitFileLoader(IDataLoader):
    """Streamlit-specific file loader for uploaded files."""
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize Streamlit file loader.
        
        Args:
            cache_enabled: Enable caching
        """
        self.csv_loader = CSVDataLoader(cache_enabled=cache_enabled)
        self._file_cache = {} if cache_enabled else None
    
    def load_file(self, uploaded_file) -> SpectralData:
        """
        Load a Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            SpectralData object
        """
        # Check cache
        if self._file_cache is not None:
            cache_key = self._get_file_hash(uploaded_file)
            if cache_key in self._file_cache:
                return self._file_cache[cache_key]
        
        # Read file content
        content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential re-reading
        
        # Parse CSV from bytes
        try:
            # Try to decode
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise FileProcessingError(f"Could not decode file: {uploaded_file.name}")
            
            # Parse CSV
            df = self._parse_csv_content(text_content, uploaded_file.name)
            
            # Extract columns
            wavelengths, absorbance = self.csv_loader._extract_columns(df)
            
            # Extract concentration
            concentration = self.csv_loader._extract_concentration(uploaded_file.name)
            
            # Create SpectralData
            data = SpectralData(
                wavelengths=wavelengths,
                absorbance=absorbance,
                filename=uploaded_file.name,
                concentration=concentration,
                metadata={'file_size': len(content)}
            )
            
            # Cache result
            if self._file_cache is not None:
                self._file_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            raise FileProcessingError(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def load_multiple(self, uploaded_files: List) -> CalibrationDataset:
        """
        Load multiple Streamlit uploaded files.
        
        Args:
            uploaded_files: List of UploadedFile objects
            
        Returns:
            CalibrationDataset object
        """
        if not uploaded_files:
            raise ValueError("No files uploaded")
        
        spectra = []
        errors = []
        
        # Progress bar
        progress_bar = st.progress(0, text="Loading files...")
        
        for i, file in enumerate(uploaded_files):
            try:
                spectrum = self.load_file(file)
                spectra.append(spectrum)
            except Exception as e:
                errors.append(f"{file.name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        
        if not spectra:
            raise FileProcessingError("Failed to load any files")
        
        if errors:
            with st.expander(f"⚠️ {len(errors)} files failed to load"):
                for error in errors:
                    st.write(f"• {error}")
        
        # Success message
        st.success(f"✅ Successfully loaded {len(spectra)} out of {len(uploaded_files)} files")
        
        return CalibrationDataset(
            spectra=spectra,
            name="Uploaded Dataset",
            metadata={
                'n_uploaded': len(uploaded_files),
                'n_loaded': len(spectra),
                'n_errors': len(errors)
            }
        )
    
    def validate_format(self, uploaded_file) -> bool:
        """
        Validate uploaded file format.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            True if valid format
        """
        try:
            # Check file extension
            name = uploaded_file.name.lower()
            if not any(name.endswith(ext) for ext in ['.csv', '.txt', '.dat']):
                return False
            
            # Check file size
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                return False
            
            # Try to load
            self.load_file(uploaded_file)
            return True
            
        except Exception:
            return False
    
    def _parse_csv_content(self, text_content: str, filename: str) -> pd.DataFrame:
        """Parse CSV content from text."""
        # Try different separators
        for sep in [',', ';', '\t', ' ']:
            try:
                df = pd.read_csv(io.StringIO(text_content), sep=sep)
                if len(df.columns) >= 2:
                    return df
            except:
                continue
        
        raise DataValidationError(f"Could not parse CSV content from {filename}")
    
    def _get_file_hash(self, uploaded_file) -> str:
        """Generate hash for uploaded file."""
        key = f"{uploaded_file.name}_{uploaded_file.size}"
        return hashlib.md5(key.encode()).hexdigest()
