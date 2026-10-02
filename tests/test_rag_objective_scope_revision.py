from rag_objective_scope_revision import scalar_slots,make_question


def test_scope_question_requests_exactly_the_retained_fields():
    slots=[dict(index=i,label=label,query_quote="old",type="number",expected=v) for i,label,v in
           ((1,"max_depth最优参数","3"),(2,"max_depth过采样后参数","5"),(3,"n_estimators最优参数","100"))]
    c=make_question(["论文.pdf"],"表5-9的LightGBM参数设置",slots)
    assert "n_estimators过采样后参数" not in c["question"]
    assert all(s["query_quote"] in c["question"] for s in c["slots"])
    assert "100" not in c["question"]


def test_scalar_filter_excludes_collections_and_propositions_keeps_model_and_date():
    c=dict(slots=[dict(index=1,label="城市",type="entity",expected="北京、上海和广州"),
       dict(index=2,label="模型",type="entity",expected="$\\varepsilon$ -SVM模型"),
       dict(index=3,label="开始",type="entity",expected="2014年5月13日"),
       dict(index=4,label="公式",type="entity",expected="V_i \\subset V_{i+1}")])
    kept=scalar_slots(c)
    assert [s["label"] for s in kept]==["模型","开始"]
    assert [s["index"] for s in kept]==[1,2]
    assert len(c["slots"])==4
