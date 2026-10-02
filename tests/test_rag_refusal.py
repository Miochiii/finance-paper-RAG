import rag_refusal as refusal


def test_contextual_refusals_are_detected():
    assert refusal.explicit_refusal("根据提供的参考上下文，无法确定该研究的参数。") == 1.0
    assert refusal.explicit_refusal("参考上下文中未提及这项数据，因此无法回答。") == 1.0
    assert refusal.explicit_refusal("抱歉，我无法找到关于该问题的证据。") == 1.0


def test_incidental_uncertainty_is_not_a_refusal():
    answer = ("动量反转包含两种效应，并可据此构建投资因子。"
              "在交易之前并无法判断每个因子体现哪一种效应。")
    assert refusal.explicit_refusal(answer) == 0.0
    assert refusal.explicit_refusal("该论文采用随机森林。📚 参考来源: [来源1]") == 0.0
