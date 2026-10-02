from rag_answer_quote_v4 import literal_anchor, presentation_map, repair_quotes, agreement_score


def test_format_markers_map_to_exact_raw_span():
    answer = "**过采样前最优参数**：`n_estimators=100`。"
    quote = "过采样前最优参数：n_estimators=100"
    anchor = literal_anchor(answer, quote)
    assert anchor["ok"]
    assert anchor["raw_quote"] == answer[anchor["start"]:anchor["end"]]
    assert presentation_map(anchor["raw_quote"])[0] == presentation_map(quote)[0]
    copied = literal_anchor(answer, "过采样前最优参数**：`n_estimators=100`")
    assert copied["ok"] and copied["mode"] == "literal"


def test_math_operators_and_subscripts_are_preserved():
    for text in ("x * y * z", "x*y*z", "x_1 + x_2", "`x*y`", "2 ** 3"):
        visible, _ = presentation_map(text)
        assert visible == text.replace(" ", "").replace("`", "")
    assert not literal_anchor("x * y = 6", "xy=6")["ok"]


def test_unique_short_entity_expands_real_clause_and_rejects_substrings():
    answer = "性能指标包括准确度、F1和AUC。下一句。"
    anchor = literal_anchor(answer, "AUC")
    assert anchor["ok"] and anchor["short_entity"]
    assert anchor["raw_quote"] == "性能指标包括准确度、F1和AUC。"
    assert not literal_anchor("AUCROC", "AUC")["ok"]
    assert not literal_anchor("AUC指标，AUC面积。", "AUC")["ok"]
    assert not literal_anchor("共有10个指标", "10")["ok"]


def test_changed_numbers_and_paraphrases_do_not_anchor():
    assert not literal_anchor("参数为100，模型为随机森林。", "参数为200")["ok"]
    assert not literal_anchor("使用AUC指标。", "使用曲线下面积AUC指标")["ok"]
    assert literal_anchor("参数为１００。", "参数为100")["ok"]


def test_repair_cannot_change_semantic_verdict_or_missing_elements():
    raw = dict(point_results=[dict(index=1,status="partial",answer_quote="虚构摘录",missing_elements=["ROC"])])
    repaired = repair_quotes(raw, dict(quotes=[dict(index=1,answer_quote="AUC",status="covered")]), [1])
    assert repaired["point_results"][0]["status"] == "partial"
    assert repaired["point_results"][0]["missing_elements"] == ["ROC"]
    assert raw["point_results"][0]["answer_quote"] == "虚构摘录"
    assert agreement_score([repaired,repaired],1,"采用AUC指标。",True)["status"] == "incomplete"
    complete = dict(point_results=[dict(index=1,status="covered",answer_quote="AUC",missing_elements=[])])
    assert agreement_score([repaired,complete],1,"采用AUC指标。",True)["status"] == "U"


def test_invalid_schema_and_covered_with_omission_keep_u():
    duplicate = dict(point_results=[dict(index=1,status="covered",answer_quote="实际回答",missing_elements=[]) for _ in range(2)])
    assert agreement_score([duplicate,duplicate],2,"实际回答",True)["status"] == "U"
    raw = dict(point_results=[dict(index=1,status="covered",answer_quote="实际回答",missing_elements=["缺失"])])
    assert agreement_score([raw,raw],1,"实际回答",True)["status"] == "U"
