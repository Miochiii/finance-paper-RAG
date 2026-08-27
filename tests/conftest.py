# -*- coding: utf-8 -*-
"""pytest 共享夹具：工作区内的临时目录（避免依赖系统 Temp，沙箱/权限无关）。"""
import os
import shutil
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

_TMP_ROOT = os.path.join(PROJECT_DIR, "tests", ".tmp")


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
