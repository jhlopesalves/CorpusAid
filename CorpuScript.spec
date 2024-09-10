# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['corpuscript.py'],
    pathex=[],
    binaries=[],
    datas=[('README.md', '.'), ('my_icon.ico', '.'), ('C:\\Users\\jhonm\\AppData\\Roaming\\nltk_data', 'nltk_data')],
    hiddenimports=['PySide6', 'bs4', 'nltk', 'chardet', 'multiprocessing'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CorpuScript',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['my_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CorpuScript',
)
