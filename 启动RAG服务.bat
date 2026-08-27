@echo off
setlocal
title RAG 一体化服务（HTTP + MCP）

cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [错误] 未找到虚拟环境: %PY%
    pause
    exit /b 1
)

echo 启动 RAG 一体化服务:
echo   HTTP API : http://127.0.0.1:8000  （/health /stats /build /search /ask /open /ingest）
echo   MCP 端点 : http://127.0.0.1:8000/mcp （供 DeepSeek Harness 注册，工具 mcp__rag__*）
echo 按 Ctrl+C 停止服务
"%PY%" -m uvicorn rag_server:app --host 127.0.0.1 --port 8000

pause
