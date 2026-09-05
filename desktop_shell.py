# -*- coding: utf-8 -*-
"""
desktop_shell.py —— 桌面壳（PyWebview 双窗口）

  主窗口  ：内嵌 DeepSeek Harness 网页（http://127.0.0.1:3080，agent 对话）
  侧栏窗口：知识库面板（本地 HTML，经 MCP 协议调用 rag 服务，与 agent 用同一服务进程）

面板功能：服务状态灯 / 启动 MCP 服务 / 打开 DSH / 文档列表 / 重建知识库。
注意：面板通过一体化服务（HTTP+MCP 同端口 8000）调用，与 agent 工具共用
同一个服务进程——不要再启动第二个会加载检索器的进程，否则争抢 qdrant 锁。

启动：python desktop_shell.py（或双击 启动桌面端.bat）
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

import webview

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DSH_URL = "http://127.0.0.1:3080"
MCP_BASE = "http://127.0.0.1:8000"   # 一体化服务：HTTP + MCP 同端口
# 源项目默认用本机 npx 缓存里的 dsh（不在 PATH）；环境变量 DSH_CMD 可覆盖
DSH_CMD = os.getenv("DSH_CMD", "dsh")

# DSH 启动日志（dsh web 会打印带 token 的访问地址，认证必需）
DSH_LOG = os.path.join(os.environ.get("TEMP", BASE), "dsh_web_start.log")
_DSH_URL_RE = re.compile(r"https?://(?:127\.0\.0\.1|localhost):3080[^\s\"']*")
_MAIN_WIN = None   # 桌面主窗口句柄（start_dsh 就绪后向其加载 DSH 页面）


def _dsh_exe():
    import shutil
    return DSH_CMD if os.path.exists(DSH_CMD) else shutil.which(DSH_CMD)


def _spawn_dsh() -> bool:
    """后台启动 dsh web（无控制台窗口，日志落盘供解析 token 地址）。"""
    exe = _dsh_exe()
    if not exe:
        return False
    try:
        os.remove(DSH_LOG)
    except OSError:
        pass
    kwargs = {"cwd": BASE,
              "stdout": open(DSH_LOG, "w", encoding="utf-8", errors="ignore"),
              "stderr": subprocess.STDOUT}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen([exe, "web", "--no-open"], **kwargs)
    return True


def _wait_dsh_url(timeout_s: int = 60) -> str:
    """轮询启动日志取带 token 的访问地址；已在运行/超时返回普通地址。"""
    for _ in range(timeout_s):
        time.sleep(1)
        try:
            with open(DSH_LOG, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            m = _DSH_URL_RE.search(txt)
            if m:
                return m.group(0)
            low = txt.lower()
            if "eaddrinuse" in low or "address already in use" in low:
                return DSH_URL   # 已在运行：无法获知旧 token，返回普通地址
        except OSError:
            pass
    return DSH_URL


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
        """手动启动 DSH：后台拉起 dsh web，解析带 token 的地址后
        自动加载进桌面主窗口（不开浏览器）；主窗口不可用时回退浏览器。"""
        if _url_ok(DSH_URL, 2.0):
            return {"ok": True, "msg": "DSH 已在运行（3080）——主窗口刷新即可；若提示需认证，请关闭 DSH 后重试本按钮"}
        if not _spawn_dsh():
            return {"ok": False, "error": f"未找到 dsh 命令（DSH_CMD={DSH_CMD}），请手动启动 DSH"}

        def _load_when_ready():
            url = _wait_dsh_url(60)
            try:
                if _MAIN_WIN is not None:
                    _MAIN_WIN.load_url(url)   # 加载到桌面主窗口，不开浏览器
                    return
            except Exception:
                pass
            import webbrowser
            webbrowser.open(url)              # 兜底：主窗口不可用才开浏览器

        threading.Thread(target=_load_when_ready, daemon=True).start()
        return {"ok": True, "msg": "DSH 启动中，就绪后自动加载到桌面主窗口（约 10~30 秒）"}

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

    def open_doc(self, doc, page=1):
        """打开知识库中某文档的原始 PDF 并跳到指定页（面板文档列表/筛选检索点击调用）。"""
        try:
            import rag_server as core
            return core.open_doc_kb(doc, int(page or 1))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- 语料管理 / 筛选检索 / 编辑器（第三步） ----
    def corpus_list(self):
        return mcp_call("corpus_list", {}, timeout=60)

    def corpus_switch(self, name):
        return mcp_call("corpus_switch", {"name": name}, timeout=60)

    def corpus_create(self, name, mineru_out):
        args = {"name": name}
        if mineru_out:
            args["mineru_out"] = mineru_out
        return mcp_call("corpus_create", args, timeout=60)

    def corpus_delete(self, name, confirm=False):
        return mcp_call("corpus_delete", {"name": name, "confirm": bool(confirm)}, timeout=120)

    def corpus_rename(self, old, new):
        return mcp_call("corpus_rename", {"old": old, "new": new}, timeout=120)

    def meta_vocab(self):
        try:
            req = urllib.request.Request(MCP_BASE + "/meta/vocab")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def survey_list(self):
        try:
            req = urllib.request.Request(MCP_BASE + "/survey/list")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def search_filters(self, query, methods=None, tasks=None, year_min=None,
                       year_max=None, author=None, top_k=8):
        args = {"query": query, "top_k": int(top_k or 8)}
        if methods:
            args["methods"] = list(methods)
        if tasks:
            args["tasks"] = list(tasks)
        if year_min:
            args["year_min"] = int(year_min)
        if year_max:
            args["year_max"] = int(year_max)
        if author:
            args["authors"] = [str(author)]
        return mcp_call("search", args, timeout=120)

    def open_url(self, url):
        import webbrowser
        try:
            webbrowser.open(url)
            return {"ok": True, "msg": "已在浏览器打开"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- 综述工作台（面板 → MCP survey 工具） ----

    def survey_outline(self, topic, outline=None, constraints=""):
        args = {"topic": topic, "constraints": constraints}
        if outline:
            args["outline"] = outline
        return mcp_call("survey_outline", args, timeout=300)

    def survey_draft(self, topic):
        return mcp_call("survey_draft", {"topic": topic}, timeout=1200)

    def survey_rewrite(self, topic, section, instruction):
        return mcp_call("survey_rewrite",
                        {"topic": topic, "section": section, "instruction": instruction},
                        timeout=600)

    def survey_edit(self, topic, section, text):
        return mcp_call("survey_edit", {"topic": topic, "section": section, "text": text},
                        timeout=120)

    def survey_section(self, topic, section):
        return mcp_call("survey_section", {"topic": topic, "section": section}, timeout=120)

    def survey_status(self, topic):
        return mcp_call("survey_status", {"topic": topic}, timeout=120)

    def survey_export(self, topic):
        return mcp_call("survey_export", {"topic": topic, "format": "markdown"}, timeout=300)


def main():
    global _MAIN_WIN
    api = Api()
    # 主窗口：内嵌 DSH agent 界面（启动后点面板「启动 DSH」，就绪自动加载进来）
    _MAIN_WIN = webview.create_window("DeepSeek Harness Agent", DSH_URL, width=1200, height=820)
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
