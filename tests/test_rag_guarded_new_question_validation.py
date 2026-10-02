from collections import defaultdict

from rag_guarded_new_question_validation import collect_exposures,exposure_intervals,canonical_candidate,evaluate_candidate
from rag_objective_slots import EXTRACT_SLOTS,CHECK_SLOTS,CANDIDATE_CHECK
from rag_objective_scope_revision import SCOPE_PROMPT


def test_historical_excerpt_attribution_does_not_treat_an_answer_as_evidence():
    source="paper.pdf";texts=defaultdict(set);questions=defaultdict(set)
    evidence="原始来源的一段完整实验正文。"*5
    collect_exposures(dict(sources=[source],question="历史问题",evidence=[dict(text=evidence)],
        answer=dict(text="模型编写的回答"*10)),{source},texts,questions)
    assert texts[source]=={evidence} and questions[source]=={"历史问题"}


def test_exclusion_spans_cover_repeated_and_whitespace_normalized_excerpts():
    text="甲乙丙丁。\n甲 乙 丙 丁。"
    intervals,unmatched=exposure_intervals(text,{"甲乙丙丁。","未找到的段落"})
    assert len(intervals)==2 and unmatched==1
    for a,b in intervals:assert "甲" in text[a:b] and "丁" in text[a:b]


def test_percent_canonical_candidate_does_not_change_value_or_original():
    candidate=dict(slots=[dict(type="number",expected="9.07%",unit="%")])
    new=canonical_candidate(candidate)
    assert new["slots"][0]["expected"]=="9.07%" and new["slots"][0]["unit"]==""
    assert candidate["slots"][0]["unit"]=="%"


def test_blind_rejection_prevents_reference_aware_check_and_is_retained():
    source="paper.pdf";text="| 持股时间 | 5个交易日 |\n| 调仓时间 | 10个交易日 |"
    slots=[dict(index=i,label=label,query_quote=label,type="number",expected=value,unit="个交易日",
                aliases=[],eid="E1",quote=quote) for i,label,value,quote in (
                    (1,"持股时间","5","| 持股时间 | 5个交易日 |"),
                    (2,"调仓时间","10","| 调仓时间 | 10个交易日 |"))]
    calls=[]
    def request(name,prompt,payload,n=0):
        calls.append(name)
        if name.startswith("candidate_"):
            assert prompt!=CANDIDATE_CHECK
            return dict(candidate=dict(question="请给出持股时间和调仓时间",slots=slots))
        if prompt==SCOPE_PROMPT:return dict(scope="表5-2的投资策略")
        if prompt==EXTRACT_SLOTS:
            assert set(payload["requested_slots"][0])=={"index","label","query_quote","type"}
            return dict(slots=[dict(index=i,supported="U",expected="",unit="",aliases=[],eid="E1",quote="") for i in (1,2)])
        assert prompt.startswith(CHECK_SLOTS)
        return dict(slot_checks=[dict(index=i,supported="U",unique="U",aliases_valid="U") for i in (1,2)])
    # evaluate_candidate reads only the fixed old-question list locally.
    class File:
        def read_text(self,encoding):return "[]"
    class Output:
        def __truediv__(self,name):assert name=="old_questions.json";return File()
    result=evaluate_candidate(Output(),"J001_1",dict(source=source,old_questions=[],evidence=[dict(eid="E1",text=text)]),request)
    assert result["valid"] and not result["qualified"] and result["blind_extraction"]
    assert not any(name.startswith("candidate_check_") for name in calls)
