# 接入 DeepSeek Harness（让 agent 调用本地 RAG）

本仓库的一体化服务（`rag_server.py`，端口 8000）已内置 MCP 端点（`http://127.0.0.1:8000/mcp`），
只需在 DSH 的 profile 补丁里注册一次，agent 即可调用本地知识库工具。

## 步骤

### 1. 启动一体化服务

```powershell
python -m uvicorn rag_server:app --host 127.0.0.1 --port 8000
```

（或使用桌面端 `desktop_shell.py` 的「▶ 启动 RAG 服务」按钮）

### 2. 编辑 DSH profile 补丁文件

打开（Windows 记事本即可）：

```
C:\Users\<你的用户名>\.dsh\profiles\web\cordis.patch.yml
```

把文件内容替换为：

```yaml
- insert:
    - id: mcp-rag
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: rag
        transport: streamable-http
        url: http://127.0.0.1:8000/mcp
        toolCallTimeoutMs: 600000
```

（缩进必须严格照抄；`toolCallTimeoutMs` 让重建类长任务不被 60 秒默认超时打断。）

### 3. 重启 DSH

补丁层只在启动时加载，重启 `dsh web` 后生效。

### 4. 验证

在 DSH 对话里问：

> 帮我看看本地知识库里有哪些文档

agent 应调用 `mcp__rag__stats` 并列出文档。可用工具：

| 工具 | 用途 |
|---|---|
| `mcp__rag__search(query, top_k)` | 检索本地知识库（返回证据块与来源，**推荐首选**） |
| `mcp__rag__stats()` | 知识库统计 + 运行统计 |
| `mcp__rag__build(chunker, clear)` | 重建知识库 |
| `mcp__rag__ingest()` | 增量入库（新文档处理后调用） |
| `mcp__rag__ask(question, top_k)` | 一键问答（检索+生成，带引用） |
| `mcp__rag__open_doc(doc, page)` | 打开某文档原始 PDF 并跳到指定页 |

## 备注

- 若 8000 端口被占用，可用环境变量 `RAG_PORT` 改端口并同步修改 `url`；
- 不要同时运行多个会加载知识库的程序（本地 qdrant 存储只允许一个进程访问）；
- 回答中的引用已带页码；如需“点击引用直接翻到 PDF 对应页”，可配合
  DSH 插件 `dsh-rag-citation`（拦截 `RAG_LINK_BASE` 链接并调起本地阅读器）。
