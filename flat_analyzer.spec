# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AstroFlatAnalyzer."""

import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None

ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")
astro_datas, astro_binaries, astro_hidden = collect_all("astropy")
dnd_datas, dnd_binaries, dnd_hidden = collect_all("tkinterdnd2")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=ctk_binaries + astro_binaries + dnd_binaries,
    datas=ctk_datas + astro_datas + dnd_datas,
    hiddenimports=ctk_hidden + astro_hidden + dnd_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroFlatAnalyzer",
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AstroFlatAnalyzer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AstroFlatAnalyzer.app",
        icon=None,
        bundle_identifier="com.acceleroto.astroflatanalyzer",
    )
