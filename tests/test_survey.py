# -*- coding: utf-8 -*-
"""交互式综述测试（文件逻辑部分，不依赖 GPU 与真实 LLM）。"""
import json
import os

import rag_core.survey as sv


def _use_tmp(monkeypatch, work_tmp):
    d = os.path.join(work_tmp, "surveys")
    monkeypatch.setattr(sv, "SURVEY_DIR", d)
    return d


def test_slugify_and_normalize():
    assert sv._slugify("机器学习在信贷风控中的应用：综述") == "机器学习在信贷风控中的应用_综述"
    secs = sv._normalize_outline(["引言", {"title": "方法综述", "keywords": ["机器学习", "风控"]}])
    assert secs[0]["title"] == "1 引言"
    assert secs[1]["title"] == "2 方法综述"  # 无编号的按序补编号
    assert secs[1]["keywords"] == ["机器学习", "风控"]


def test_draft_parts_roundtrip():
    text = "## 1 引言\n\n正文一。\n\n## 2 方法\n\n正文二。"
    parts = sv._draft_parts(text)
    assert len(parts) == 2 and parts[0]["body"] == "正文一。"
    dumped = sv._dump_draft(parts, "主题")
    assert "## 1 引言" in dumped and "正文二。" in dumped


def test_find_section():
    parts = [{"title": "1 引言", "body": "a"}, {"title": "2 方法", "body": "b"}]
    assert sv._find_section(parts, "方法") == 1
    assert sv._find_section(parts, 1) == 0
    assert sv._find_section(parts, "不存在") is None


def test_outline_manual_save_and_status(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    r = sv.survey_outline("测试主题", outline=[{"title": "1 引言", "keywords": ["a"]}])
    assert r["ok"] and r["mode"] == "manual"
    st = sv.survey_status("测试主题")
    assert st["outline"] and st["sections"] == []  # 草稿未生成


def test_draft_edit_section_export(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: f"这是{u.split('本节：')[1].split(chr(10))[0]}的正文[来源1]。")
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [
        {"source": "a.pdf", "page": 3, "author": "甲", "year": 2022, "text": "证据A"}])

    sv.survey_outline("主题甲", outline=[{"title": "1 引言", "keywords": []},
                                        {"title": "2 方法", "keywords": []}])
    r = sv.survey_draft("主题甲")
    assert r["ok"] and r["written"] == ["1 引言", "2 方法"]
    assert r["refs"] == 2
    # 节末自动附“本节引用”注（人读文献名+年份+页码）
    sec = sv.survey_section("主题甲", "引言")
    assert "【本节引用】" in sec["body"] and "[来源1] 甲. a（2022），第3页" in sec["body"]
    # 人读对照表 refs.md 已生成
    assert os.path.isfile(os.path.join(sv._survey_dir("主题甲"), "refs.md"))
    # 断点续写：再次 draft 不重复生成
    r2 = sv.survey_draft("主题甲")
    assert r2["written"] == []
    # 手动编辑（无引用 → 不追加注记）
    assert sv.survey_edit("主题甲", "引言", "手动改的引言。")["ok"]
    sec = sv.survey_section("主题甲", "引言")
    assert sec["body"] == "手动改的引言。"
    # 手动编辑（含引用 → 自动补注记）
    assert sv.survey_edit("主题甲", "引言", "手动改的引言[来源2]。")["ok"]
    sec = sv.survey_section("主题甲", "引言")
    assert "【本节引用】" in sec["body"] and "[来源2]" in sec["body"]
    # 导出：注记剥离 + 引用重编号 + 参考文献
    e = sv.survey_export("主题甲")
    assert e["ok"]
    out = open(e["path"], encoding="utf-8").read()
    assert "【本节引用】" not in out
    assert "[来源1]" not in out
    assert "[1] 甲. a（2022），第 3 页" in out
    assert "## 参考文献" in out


def test_editor_data_auto_export_and_stable(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: "正文[来源1]。")
    sv.survey_outline("主题丁", outline=[{"title": "1 引言", "keywords": []}])
    sv.survey_draft("主题丁")
    r = sv.survey_editor_data("主题丁")
    assert r["ok"] is True
    assert "引言" in r["text"] and "## 参考文献" in r["text"]
    assert os.path.isfile(r["path"])
    # 二次调用：导出文件已存在，内容稳定不重复导出
    r2 = sv.survey_editor_data("主题丁")
    assert r2["text"] == r["text"]


def test_editor_data_no_draft(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    r = sv.survey_editor_data("没有草稿的主题")
    assert r["ok"] is False and "草稿" in r.get("error", "")


def _setup_two_section_draft(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])

    def fake_draft_llm(system, user):
        sec = user.split("本节：")[1].split(chr(10))[0]
        if "2 方法" in sec:
            return "这是2 方法的第一段[来源1]。\n\n这是2 方法的第二段[来源1]。"
        return "这是" + sec + "的正文[来源1]。"

    monkeypatch.setattr(sv, "_llm_text", fake_draft_llm)
    sv.survey_outline("主题戊", outline=[{"title": "1 引言", "keywords": []},
                                        {"title": "2 方法", "keywords": []}])
    sv.survey_draft("主题戊")
    return sv


def test_rewrite_selection_v1_section_context(monkeypatch, work_tmp):
    _setup_two_section_draft(monkeypatch, work_tmp)
    captured = {}
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: captured.update(user=u) or "改写后的段落。")
    sel = "这是2 方法的第一段[1]。"
    r = sv.survey_rewrite_selection("主题戊", sel, "改得更学术化")
    assert r["ok"] and r["rewritten_text"] == "改写后的段落。"
    assert r["evidence"] == [] and r["context_scope"] == "section"
    u = captured["user"]
    assert "待改写段落" in u and sel in u
    assert "第二段" in u            # 本节其它段落进上下文
    assert u.count("第一段") == 1   # 选中段只在"待改写段落"出现一次
    assert "引言的正文" not in u    # 本节上下文不带第1节内容


def test_rewrite_selection_full_scope(monkeypatch, work_tmp):
    _setup_two_section_draft(monkeypatch, work_tmp)
    captured = {}
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: captured.update(user=u) or "全文语境改写。")
    sel = "这是2 方法的第一段[1]。"
    r = sv.survey_rewrite_selection("主题戊", sel, "扩写", context_scope="full")
    assert r["ok"] and r["context_scope"] == "full"
    assert "引言的正文" in captured["user"]   # 全文上下文包含第1节内容


def test_rewrite_selection_evidence_mode(monkeypatch, work_tmp):
    _setup_two_section_draft(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [
        {"source": "e.pdf", "page": 9, "author": "丙", "year": 2024, "text": "关键证据内容"}])
    captured = {}
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: captured.update(user=u) or "基于证据的改写。")
    sel = "这是2 方法的第一段[1]。"
    r = sv.survey_rewrite_selection("主题戊", sel, "补充论据", evidence=True)
    assert r["ok"] and r["rewritten_text"] == "基于证据的改写。"
    assert len(r["evidence"]) == 1 and r["evidence"][0]["source"] == "e.pdf"
    assert "关键证据内容" in captured["user"]


def test_rewrite_selection_validation(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    # 无草稿：不调 LLM 直接报错
    r = sv.survey_rewrite_selection("没有草稿的主题", "足够长的一段选中文字内容")
    assert r["ok"] is False and "草稿" in r.get("error", "")
    # 选中文字过短
    sv.survey_outline("主题己", outline=[{"title": "1 引言", "keywords": []}])
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: "正文[来源1]。")
    sv.survey_draft("主题己")
    r2 = sv.survey_rewrite_selection("主题己", "太短")
    assert r2["ok"] is False and "太短" in r2.get("error", "")


def test_editor_save_default_and_saveas(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: "正文[来源1]。")
    sv.survey_outline("主题庚", outline=[{"title": "1 引言", "keywords": []}])
    sv.survey_draft("主题庚")
    # 默认保存：覆盖 exports/<slug>.markdown
    r = sv.survey_editor_save("主题庚", "修改后的正文。")
    assert r["ok"] and r["filename"] == "主题庚.markdown"
    assert open(r["path"], encoding="utf-8").read() == "修改后的正文。"
    # 另存为：新文件，不覆盖默认文件
    r2 = sv.survey_editor_save("主题庚", "第二版。", filename="修改稿 v2.md")
    assert r2["ok"] and r2["filename"] == "修改稿 v2.md"
    assert open(r2["path"], encoding="utf-8").read() == "第二版。"
    assert open(r["path"], encoding="utf-8").read() == "修改后的正文。"


def test_editor_save_filename_whitelist(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])
    monkeypatch.setattr(sv, "_llm_text", lambda s, u: "正文[来源1]。")
    sv.survey_outline("主题辛", outline=[{"title": "1 引言", "keywords": []}])
    sv.survey_draft("主题辛")
    for bad in ["../穿越.md", "..\\穿越.md", "a/b.md", "a.txt", "a"]:
        r = sv.survey_editor_save("主题辛", "x", filename=bad)
        assert r["ok"] is False and "文件名" in r.get("error", ""), bad
    d = os.path.join(sv._survey_dir("主题辛"), "exports")
    names = os.listdir(d) if os.path.isdir(d) else []
    assert all("穿越" not in n for n in names)


def test_editor_save_no_draft(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    r = sv.survey_editor_save("没有草稿的主题", "x")
    assert r["ok"] is False and "草稿" in r.get("error", "")


def test_draft_resume_and_rewrite(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [
        {"source": "b.pdf", "page": 5, "author": "乙", "year": 2023, "text": "证据B"}])
    seen = []

    def fake_llm(s, u):
        if "修改指令" in u:
            seen.append(u)
            return "重写后的正文[来源1]。"
        return "普通正文[来源1]。"

    monkeypatch.setattr(sv, "_llm_text", fake_llm)

    sv.survey_outline("主题乙", outline=[{"title": "1 引言", "keywords": []}])
    sv.survey_draft("主题乙")
    r = sv.survey_rewrite("主题乙", "引言", "更深入")
    assert r["ok"] and "重写后的正文[来源1]。" in r["body"]
    assert "【本节引用】" in r["body"]
    # 旧引用注不进入重写 prompt（避免注记污染 LLM 输入）
    assert all("【本节引用】" not in u for u in seen)
    refs = sv._load_refs(sv._survey_dir("主题乙"))
    assert len(refs) >= 1


def test_draft_force_replaces_not_duplicates(monkeypatch, work_tmp):
    _use_tmp(monkeypatch, work_tmp)
    monkeypatch.setattr(sv, "_collect_evidence", lambda q, top_k=15: [])
    calls = {"n": 0}

    def fake_llm(s, u):
        calls["n"] += 1
        return f"第{calls['n']}版正文。"

    monkeypatch.setattr(sv, "_llm_text", fake_llm)

    sv.survey_outline("主题丙", outline=[{"title": "1 引言", "keywords": []}])
    sv.survey_draft("主题丙")
    sv.survey_draft("主题丙", force=True)
    txt = open(os.path.join(sv._survey_dir("主题丙"), "draft.md"), encoding="utf-8").read()
    assert txt.count("## 1 引言") == 1, "force 重写不应重复追加旧节"
    assert "第2版正文" in txt and "第1版正文" not in txt
