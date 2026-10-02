from rag_prospective_objective_validation import choose_cards,question_signature
from rag_fresh_change_experiment import sha


def test_cards_are_bounded_nonoverlapping_original_tables():
    table="[TABLE_START]\n表5-1 样本参数描述\n| 模型 | 数值A | 数值B |\n| --- | --- | --- |\n| 模型甲 | 32 | 64 |\n| 模型乙 | 16 | 128 |\n[/TABLE_END]"
    text="简介。"+table+"其他说明。"+table.replace("5-1","5-2")
    cards=choose_cards(text,"x.pdf")
    assert len(cards)==2
    for c in cards:
        assert c["text"]==text[c["start"]:c["end"]]
        assert c["text_sha256"]==sha(c["text"]) and len(c["text"])<=900
    assert max(0,min(cards[0]["end"],cards[1]["end"])-max(cards[0]["start"],cards[1]["start"]))==0


def test_complete_request_signature_keeps_different_scopes():
    a="依据论文《甲》，针对表5-1，请仅逐项给出以下字段：1）样本数。"
    b="依据论文《乙》，针对表5-1，请仅逐项给出以下字段：1）样本数。"
    assert question_signature(a)==question_signature(b)
    assert question_signature(a)!=question_signature(a.replace("5-1","5-2"))
