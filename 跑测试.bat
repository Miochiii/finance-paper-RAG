@echo off
setlocal
title RAG 单元测试（pytest）

cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [错误] 未找到虚拟环境: %PY%
    pause
    exit /b 1
)

echo 运行单元测试（不需要 GPU / 知识库服务）...
"%PY%" -m pytest tests -q
echo.
pause
