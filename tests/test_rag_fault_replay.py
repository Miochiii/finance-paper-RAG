import pytest

from rag_fault_replay import gold_hit, remove_gold_evidence, select_cases


def test_remove_gold_preserves_context_count_and_excludes_gold():
    base = [{"source": "gold.pdf", "text": "answer"},
            {"source": "other.pdf", "text": "other"}]
    donors = [{"source": "gold.pdf", "text": "duplicate answer"},
              {"source": "third.pdf", "text": "distractor"}]
    fault = remove_gold_evidence(base, {"gold.pdf"}, donors)
    assert len(fault) == len(base)
    assert gold_hit(base, {"gold.pdf"}) == 1
    assert gold_hit(fault, {"gold.pdf"}) == 0
    assert {e["source"] for e in fault} == {"other.pdf", "third.pdf"}


def test_remove_gold_rejects_insufficient_donors():
    with pytest.raises(ValueError, match="异源证据"):
        remove_gold_evidence([{"source": "gold.pdf", "text": "answer"}],
                             {"gold.pdf"}, [])


def test_case_selection_requires_question_gold_and_good_existing_answer():
    rows = [{"qid": "a", "question": "q", "gold_answer": "a",
             "gold_sources": "gold.pdf", "judge_corr": "4"},
            {"qid": "b", "question": "q", "gold_answer": "a",
             "gold_sources": "gold.pdf", "judge_corr": "2"}]
    cache = {"a": [{"source": "gold.pdf", "text": "x"}],
             "b": [{"source": "gold.pdf", "text": "x"}]}
    assert len(select_cases(rows, cache, 2)) == 1
