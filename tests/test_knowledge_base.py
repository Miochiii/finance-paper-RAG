# -*- coding: utf-8 -*-
"""知识库条目构造 / 去重合并 / 元数据透传 测试。"""
import os

from rag_core.knowledge_base import KnowledgeBase, _chunk_entries


def test_chunk_entries_metadata(work_tmp):
    es = _chunk_entries(["t1", "t2"], "s.pdf", "hmm", "P:/s.pdf", [(3, 4), (None, None)])
    assert es[0]["page_start"] == 3 and es[0]["page_end"] == 4
    assert es[0]["pdf_path"] == "P:/s.pdf"
    assert "page_start" not in es[1]
    assert len(es) == 2


def test_add_entries_and_save(work_tmp):
    store = os.path.join(work_tmp, "kb.json")
    kb = KnowledgeBase(store)
    kb.add_entries(_chunk_entries(["文本甲"], "a.pdf", "hmm", "P:/a.pdf", [(1, 1)]))
    kb.save()
    assert os.path.exists(store)
    loaded = KnowledgeBase(store).load()
    assert len(loaded) == 1 and loaded[0]["source"] == "a.pdf"


def test_save_dedup_and_metadata_merge(work_tmp):
    store = os.path.join(work_tmp, "kb.json")
    kb = KnowledgeBase(store)
    kb.add_entries(_chunk_entries(["同一文本"], "a.pdf", "hmm", None, None))
    kb.save()
    # 再次写入同文本但带页码元数据：应合并字段而非重复追加
    kb2 = KnowledgeBase(store)
    kb2.add_entries(_chunk_entries(["同一文本"], "a.pdf", "hmm", "P:/a.pdf", [(7, 7)]))
    kb2.save()
    loaded = KnowledgeBase(store).load()
    assert len(loaded) == 1
    assert loaded[0]["page_start"] == 7 and loaded[0]["pdf_path"] == "P:/a.pdf"


def test_to_texts_metadatas_fields(work_tmp):
    store = os.path.join(work_tmp, "kb.json")
    kb = KnowledgeBase(store)
    kb.add_entries(_chunk_entries(["t1"], "a.pdf", "hmm", "P:/a.pdf", [(2, 5)]))
    kb.save()
    texts, metas = KnowledgeBase(store).to_texts_metadatas()
    assert texts == ["t1"]
    assert metas[0]["page_start"] == 2 and metas[0]["page_end"] == 5
    assert metas[0]["pdf_path"] == "P:/a.pdf"
