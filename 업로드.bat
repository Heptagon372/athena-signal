@echo off
REM ---------------------------------------------------------------
REM  GitHub upload launcher.
REM  All logic lives in upload.ps1 -- cmd.exe mis-parses batch files
REM  that mix UTF-8 Korean with parenthesised blocks, so this wrapper
REM  is deliberately ASCII-only.
REM ---------------------------------------------------------------
chcp 65001 > nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload.ps1"
