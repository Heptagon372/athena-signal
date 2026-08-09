@echo off
REM ---------------------------------------------------------------
REM  Athena Signal launcher -- the only .bat left.
REM  Everything else is a menu item inside athena.py.
REM
REM  ASCII only, on purpose: cmd.exe reads batch files by byte
REM  offset, so Korean text in a chcp'd .bat splits commands in half.
REM ---------------------------------------------------------------
chcp 65001 > nul
cd /d "%~dp0"
title Athena Signal

python athena.py
if errorlevel 1 (
    echo.
    echo   Could not start. Is Python installed?  https://python.org
    echo.
    pause
)
