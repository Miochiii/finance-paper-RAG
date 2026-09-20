# -*- coding: utf-8 -*-
"""pytest 共享夹具：工作区内的临时目录（避免依赖系统 Temp，沙箱/权限无关）。"""
import os
import shutil
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

_TMP_ROOT = os.path.join(PROJECT_DIR, "tests", ".tmp")

# 测试隔离：部分测试会真实调用 search_kb/ask_kb，若不改指向就会污染生产数据——
#   1) 观测日志（jsonl）写到 tests/.tmp，不污染 data/observability.jsonl；
#   2) 结构化分析层写向测试库 rag_analytics_test（通常不存在 → 实时落库静默失败），
#      绝不写进真实的 rag_analytics。
import rag_core.config as cfg          # noqa: E402
import rag_core.observability as obs   # noqa: E402

cfg.MYSQL_DB = "rag_analytics_test"
obs.OBS_LOG_FILE = os.path.join(_TMP_ROOT, "observability_test.jsonl")


@pytest.fixture(scope="session")
def work_tmp():
    """会话级临时目录（tests/.tmp，用完清理）。"""
    os.makedirs(_TMP_ROOT, exist_ok=True)
    d = os.path.join(_TMP_ROOT, "t")
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
