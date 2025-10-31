# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Project root (spec is executed, __file__ may be undefined in some contexts)
ROOT = Path.cwd()

# Entry point
entry_script = str(ROOT / 'qt_app' / 'main_qt.py')

# Collect data files for key libs
datas = []
datas += collect_data_files('matplotlib', include_py_files=True)
# Rely on PyInstaller hooks for sklearn; avoid bundling its tests
try:
    datas += collect_data_files('xgboost', include_py_files=True)
except Exception:
    pass
datas += collect_data_files('calibration', include_py_files=True)

# Hidden imports to ensure dynamic modules are included
hiddenimports = []
hiddenimports += collect_submodules('matplotlib')
# Avoid pulling entire sklearn test tree; default hooks handle runtime needs
try:
    hiddenimports += collect_submodules('xgboost')
except Exception:
    pass
hiddenimports += [
    'PyQt6',
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.QtSvg',
]

block_cipher = None


excludes = [
    # Heavy/unused ML stacks that caused crashes during collection
    'torch', 'torchvision', 'torchaudio', 'functorch',
    'onnx', 'onnxruntime',
    'lightgbm',
    # Web/vis stacks not needed for the Qt exe
    'streamlit', 'plotly', 'altair', 'pydeck', 'tornado', 'watchdog',
    # Dev/test frameworks pulled in by sklearn hooks
    'pytest', 'sympy', 'numba', 'h5py', 'sqlalchemy', 'pyarrow', 'skopt',
    # ONNX conversion stack not used by the Qt app
    'skl2onnx',
    # Optional pandas writers and extras not needed
    'openpyxl', 'xlsxwriter', 'pandas.io.formats.style', 'pandas.io.clipboard',
    # Matplotlib tests/docs and non-Qt backends
    'matplotlib.tests', 'matplotlib.testing', 'matplotlib.sphinxext',
    'tkinter', 'matplotlib.backends._backend_tk', 'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_tkcairo', 'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_pgf',
    'matplotlib.backends.backend_ps', 'matplotlib.backends.backend_svg', 'matplotlib.backends.backend_template',
    'matplotlib.backends.backend_gtk3', 'matplotlib.backends.backend_gtk4', 'matplotlib.backends.backend_wx',
]

a = Analysis(
    [entry_script],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['QtAgg']}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectralAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # set to 'assets/app.ico' if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpectralAnalysis'
)


