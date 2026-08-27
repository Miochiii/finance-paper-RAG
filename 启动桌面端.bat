@echo off
setlocal
title RAG 桌面端（DSH Agent + 知识库面板）

cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [错误] 未找到虚拟环境: %PY%
    pause
    exit /b 1
)

"%PY%" desktop_shell.py

pause
