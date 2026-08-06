# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_server.py'],
    pathex=[],
    binaries=[],
    datas=[('web', 'web')],
    hiddenimports=['uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'bs4', 'feedparser', 'yfinance', 'pandas', 'numpy', 'requests', 'data_sources.symbol_registry', 'data_sources.kr_market', 'data_sources.kis_client', 'data_sources.market_clock', 'data_sources.http_client', 'data_sources.price_provider', 'data_sources.news_crawler', 'data_sources.community_crawler', 'engine.indicators', 'engine.scoring', 'engine.backtest', 'storage.db'],
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
    a.binaries,
    a.datas,
    [],
    name='AthenaSignal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
