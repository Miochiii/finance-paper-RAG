from rag_candidate_context_revision import evaluate_candidate,NEW_CANDIDATE_PROMPT


def test_old_questions_never_enter_creation_request_but_local_row_is_preserved():
    calls=[]
    row=dict(source="paper.pdf",evidence=[dict(eid="E1",text="原文")],old_questions=["不应传入的旧题"])
    def request(name,prompt,payload,n=0):
        calls.append((name,prompt,payload))
        return dict(candidate=None,reason="不能出题")
    result=evaluate_candidate(None,"J001_1",row,request)
    assert not result["qualified"] and row["old_questions"]==["不应传入的旧题"]
    assert calls[0][1]==NEW_CANDIDATE_PROMPT
    assert set(calls[0][2])=={"paper_sources","mineru_evidence"}
