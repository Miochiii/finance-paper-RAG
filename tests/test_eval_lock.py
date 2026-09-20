# -*- coding: utf-8 -*-
"""评测互斥与进程存活探测的测试。

守两条线：
  1. **存活探测绝不能杀掉目标进程**——Windows 上 os.kill(pid, 0) 是 TerminateProcess
     的封装，曾把正在运行的评测杀掉（真实事故），所以这里专门做回归测试；
  2. 互斥语义：持有者存活 → 拒绝启动；陈旧锁 → 自动接管；释放后锁文件消失。
测试全部在 work_tmp 下改指向 OUTPUT_DIR，不碰真实 results/。
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate as ev


class TestPidAlive:
    def test_self_is_alive(self):
        assert ev._pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_dead(self):
        assert ev._pid_alive(999999) is False

    def test_invalid_pid(self):
        assert ev._pid_alive(0) is False
        assert ev._pid_alive(-1) is False

    def test_probe_does_not_kill_target(self):
        """回归：探测后目标进程必须仍然活着（旧实现 os.kill(pid,0) 会杀掉它）。"""
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        try:
            assert ev._pid_alive(p.pid) is True
            assert p.poll() is None, "探测把目标进程杀掉了——绝不能再用 os.kill 探测"
        finally:
            p.kill()
            p.wait(timeout=10)


class TestEvalRunLock:
    @staticmethod
    def _use_tmp(monkeypatch, tmp):
        monkeypatch.setattr(ev, "OUTPUT_DIR", str(tmp))
        return os.path.join(str(tmp), ".eval.lock")

    def test_acquire_then_refuse_same_process(self, monkeypatch, work_tmp):
        # 每个用例用独立子目录，避免彼此干扰
        d = os.path.join(work_tmp, "lock_a")
        os.makedirs(d, exist_ok=True)
        lock = self._use_tmp(monkeypatch, d)
        assert ev._eval_run_lock() is None                 # 首次获取成功
        assert os.path.exists(lock)
        info = json.load(open(lock, encoding="utf-8"))
        assert info["pid"] == os.getpid()                  # 记录的是当前进程
        busy = ev._eval_run_lock()                         # 同一进程仍存活 → 拒绝
        assert busy and "已有评测在运行" in busy
        ev._release_eval_lock()
        assert not os.path.exists(lock)

    def test_stale_lock_is_taken_over(self, monkeypatch, work_tmp):
        d = os.path.join(work_tmp, "lock_b")
        os.makedirs(d, exist_ok=True)
        lock = self._use_tmp(monkeypatch, d)
        with open(lock, "w", encoding="utf-8") as f:
            json.dump({"pid": 999999, "started": "stale"}, f)
        assert ev._eval_run_lock() is None                 # 陈旧锁 → 接管
        assert json.load(open(lock, encoding="utf-8"))["pid"] == os.getpid()
        ev._release_eval_lock()

    def test_corrupt_lock_is_taken_over(self, monkeypatch, work_tmp):
        d = os.path.join(work_tmp, "lock_c")
        os.makedirs(d, exist_ok=True)
        lock = self._use_tmp(monkeypatch, d)
        with open(lock, "w", encoding="utf-8") as f:
            f.write("这不是 JSON")
        assert ev._eval_run_lock() is None
        ev._release_eval_lock()

    def test_release_is_idempotent(self, monkeypatch, work_tmp):
        d = os.path.join(work_tmp, "lock_d")
        os.makedirs(d, exist_ok=True)
        self._use_tmp(monkeypatch, d)
        ev._release_eval_lock()                            # 不存在也不该抛
        ev._eval_run_lock()
        ev._release_eval_lock()
        ev._release_eval_lock()
