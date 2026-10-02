from diagnose_mineru_evidence_coverage import make_chunks,literal_coverage,audit_records
import pytest


def test_overlapping_chunks_preserve_exact_original_spans():
    text='甲乙丙丁'*100
    chunks=make_chunks(text,'a.pdf',size=150,stride=100)
    assert chunks
    assert all(text[c['start']:c['end']]==c['text'] for c in chunks)
    audit_records({'q':dict(actual_visible_sources=['a.pdf'],retrieved=chunks)},[dict(source='a.pdf',text=text)])


def test_quote_match_is_literal_not_synonym_and_scope_mismatch_fails():
    assert literal_coverage([dict(quote='模型构建')],['模型\n构建。'])==[True]
    assert literal_coverage([dict(quote='模型构建')],['建立模型。'])==[False]
    with pytest.raises(ValueError):audit_records({'q':dict(actual_visible_sources=['b.pdf'],retrieved=[dict(source='a.pdf',start=0,end=3,text='原文。')])},[dict(source='a.pdf',text='原文。')])
