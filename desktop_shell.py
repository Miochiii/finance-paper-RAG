# -*- coding: utf-8 -*-
"""
desktop_shell.py —— 桌面壳（PyWebview 双窗口）

  主窗口  ：内嵌 DeepSeek Harness 网页（http://127.0.0.1:3080，agent 对话）
  侧栏窗口：知识库面板（本地 HTML，经 MCP 协议调用 rag 服务，与 agent 用同一服务进程）

面板功能：服务状态灯 / 启动 MCP 服务 / 打开 DSH / 文档列表 / 重建知识库。
注意：面板通过 MCP(8001) 调用，与 agent 工具共用同一个服务进程——不要同时再开
HTTP 服务（rag_server.py 的 8000 端口），否则会争抢 qdrant 本地存储锁。

启动：python desktop_shell.py（或双击 启动桌面端.bat）
"""

import json
import os
import subprocess
import sys
import threading
import urllib.request

import webview

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DSH_URL = "http://127.0.0.1:3080"
MCP_BASE = "http://127.0.0.1:8000"   # 一体化服务：HTTP + MCP 同端口
# 默认直接调用 PATH 里的 dsh；未全局安装时在 .env 设 DSH_CMD=完整路径（如 ...\dsh.cmd）
DSH_CMD = os.getenv("DSH_CMD", "dsh")


# --------------------------------------------------------------------------
# MCP streamable-http 客户端（仅标准库；与 DSH 的 dsh-mcp-client 同一协议）
# --------------------------------------------------------------------------
def mcp_call(tool: str, arguments: dict, timeout: int = 60) -> dict:
    """调用 MCP 工具并返回其 JSON 结果。"""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    def post(payload, extra=None):
        h = dict(headers)
        if extra:
            h.update(extra)
        req = urllib.request.Request(
            MCP_BASE + "/mcp",
            data=json.dumps(payload).encode("utf-8"),
            headers=h,
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read().decode("utf-8"), resp.headers.get("Mcp-Session-Id")

    # 握手：initialize → 拿到会话 id
    try:
        _, sid = post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "desktop-shell", "version": "1.0"},
            },
        })
    except Exception as e:
        return {"ok": False, "error": f"MCP 服务未启动或连接失败: {e}"}

    # 调用工具
    try:
        body, _ = post(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": tool, "arguments": arguments}},
            {"Mcp-Session-Id": sid} if sid else None,
            # 注意：post 的 timeout 参数在 urlopen 处；这里直接传全局 timeout
        )
    except Exception as e:
        return {"ok": False, "error": f"调用 {tool} 失败: {e}"}

    # 解析 SSE：取 data: 行
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except Exception:
            continue
        if payload.get("error"):
            return {"ok": False, "error": str(payload["error"])}
        result = payload.get("result") or {}
        content = result.get("content") or []
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        if texts:
            try:
                return json.loads(texts[0])
            except Exception:
                return {"ok": True, "raw": texts[0]}
        return {"ok": True, "raw": result}
    return {"ok": False, "error": "无响应数据"}


def _url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 面板 JS API（panel.html 通过 pywebview.api.* 调用）
# --------------------------------------------------------------------------
class Api:
    def __init__(self):
        self._last_build = None

    def get_status(self):
        kb = mcp_call("stats", {}, timeout=10)
        return {
            "dsh": _url_ok(DSH_URL, 2.0),
            "kb": kb,
        }

    def start_mcp(self):
        bat = os.path.join(BASE, "启动RAG服务.bat")
        if os.path.exists(bat):
            subprocess.Popen(["cmd", "/c", bat], cwd=BASE)
            return {"ok": True, "msg": "RAG 一体化服务启动中（8000，HTTP+MCP），几秒后点刷新"}
        return {"ok": False, "error": f"未找到 {bat}"}

    def start_dsh(self):
        if os.path.exists(DSH_CMD):
            subprocess.Popen([DSH_CMD, "web", "--no-open"])
            return {"ok": True, "msg": "DSH 启动中，约 10~30 秒后点刷新"}
        return {"ok": False, "error": f"未找到 dsh 命令（DSH_CMD={DSH_CMD}），请手动启动 DSH"}

    def rebuild(self, chunker="hmm"):
        def _run():
            self._last_build = mcp_call(
                "build", {"chunker": chunker, "clear": False}, timeout=1800
            )
        self._last_build = None
        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "msg": "重建已开始（约 2~5 分钟，含向量重建），完成后点刷新"}

    def ingest(self):
        def _run():
            self._last_build = mcp_call("ingest", {}, timeout=1800)
        self._last_build = None
        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "msg": "增量更新已开始（新增文档秒级~分钟级），完成后点刷新"}

    def last_build(self):
        return self._last_build if self._last_build is not None else {"ok": None, "msg": "尚无重建任务"}

    def open_doc(self, doc):
        """打开知识库中某文档的原始 PDF（面板文档列表点击调用）。"""
        try:
            import rag_server as core
            return core.open_doc_kb(doc, 1)
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    api = Api()
    # 主窗口：内嵌 DSH agent 界面
    webview.create_window("DeepSeek Harness Agent", DSH_URL, width=1200, height=820)
    # 侧栏窗口：知识库面板
    webview.create_window(
        "知识库面板",
        os.path.join(BASE, "panel.html"),
        js_api=api,
        width=440,
        height=760,
        x=1230,
        y=60,
    )
    webview.start()


if __name__ == "__main__":
    main()
