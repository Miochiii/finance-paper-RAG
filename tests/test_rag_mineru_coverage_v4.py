from rag_mineru_coverage_v4 import evidence_projection, source_context, point_quality, summarize
from rag_answer_quote_v4 import agreement_score


def test_proxy_projection_removes_offline_gold_and_metadata():
    rows=[dict(text="真实原文",source="a.pdf",point="gold",qid="private",rerank={})]
    assert evidence_projection(rows)==[dict(eid="C0001",text="真实原文")]
    assert source_context(rows)=="[来源1]（来源: a.pdf）\n真实原文"


def test_quality_total_is_derived_from_validated_points():
    def result(status):
        raw=dict(point_results=[dict(index=1,status=status,answer_quote="实际回答",missing_elements=[])])
        return dict(signal=agreement_score([raw,raw],1,"实际回答",True))
    assert point_quality(result("covered"))=="2"
    assert point_quality(result("partial"))=="1"
    assert point_quality(result("missing"))=="0"
    assert point_quality(result("contradicted"))=="0"
    assert point_quality(result("U"))=="U"
    assert point_quality(result("covered"),"length")=="U"


def test_u_remains_in_metrics_and_is_not_a_success():
    rows=[dict(quality="1",proxy_status="U",rule_refusal=0),
          dict(quality="2",proxy_status="incomplete",rule_refusal=0),
          dict(quality="U",proxy_status="complete",rule_refusal=0)]
    m=summarize(rows)
    assert m["n"]==3 and m["quality_u"]==1 and m["proxy_u"]==1
    assert m["unresolved_incomplete"]==1 and m["false_flags"]==1 and m["flagged_incomplete"]==0
