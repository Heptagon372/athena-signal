@echo off
echo ===================================================
echo   Athena Signal - Windows EXE Builder
echo ===================================================
echo.
echo Run this file only once.
echo When finished, AthenaSignal.exe will appear in the "dist" folder.
echo.

python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt

echo.
echo [Building... this can take 1-2 minutes]
echo.

python -m PyInstaller --noconfirm --onefile --console ^
  --name AthenaSignal ^
  --add-data "web;web" ^
  --hidden-import uvicorn.logging ^
  --hidden-import uvicorn.loops ^
  --hidden-import uvicorn.loops.auto ^
  --hidden-import uvicorn.protocols ^
  --hidden-import uvicorn.protocols.http ^
  --hidden-import uvicorn.protocols.http.auto ^
  --hidden-import uvicorn.protocols.websockets ^
  --hidden-import uvicorn.protocols.websockets.auto ^
  --hidden-import uvicorn.lifespan ^
  --hidden-import uvicorn.lifespan.on ^
  --hidden-import bs4 ^
  --hidden-import feedparser ^
  --hidden-import yfinance ^
  --hidden-import pandas ^
  --hidden-import numpy ^
  --hidden-import requests ^
  --hidden-import data_sources.symbol_registry ^
  --hidden-import data_sources.kr_market ^
  --hidden-import data_sources.kis_client ^
  --hidden-import data_sources.market_clock ^
  --hidden-import data_sources.http_client ^
  --hidden-import data_sources.price_provider ^
  --hidden-import data_sources.news_crawler ^
  --hidden-import data_sources.community_crawler ^
  --hidden-import engine.indicators ^
  --hidden-import engine.scoring ^
  --hidden-import engine.backtest ^
  --hidden-import storage.db ^
  run_server.py

echo.
if exist dist\AthenaSignal.exe (
    echo [SUCCESS] dist\AthenaSignal.exe has been created.
    echo From now on, just double-click that exe file.
    echo It will start the server and open the dashboard in your browser automatically.
) else (
    echo [FAILED] The exe was not created.
    echo Please copy the error messages shown above and share them.
)
echo.
pause
