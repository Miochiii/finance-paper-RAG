from rag_placebo_control import make_placebo, prompt_sources


def test_placebo_preserves_gold_and_replaces_matching_count():
    base = [
        {"source": "gold.pdf", "text": "正确片段"},
        {"source": "other.pdf", "text": "无关甲"},
        {"source": "other.pdf", "text": "无关乙"},
    ]
    donors = [{"source": "new.pdf", "text": "替换片段"}]
    result, indices, sources = make_placebo(base, {"gold.pdf"}, donors, qid="q1")
    assert len(result) == len(base) == 3
    assert result[0] == base[0]
    assert indices == [1] and sources == ["new.pdf"]
    assert result[1] != base[1]
    assert make_placebo(base[:1], {"gold.pdf"}, donors) is None


def test_prompt_sources_reads_actual_context_headers():
    context = "[来源1]（来源: gold.pdf）\n内容\n\n[来源2]（来源: other.pdf）\n内容"
    assert prompt_sources(context) == ["gold.pdf", "other.pdf"]
