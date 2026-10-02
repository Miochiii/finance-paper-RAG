import csv
from pathlib import Path

from prepare_rag_blind_review import prepare


def test_blind_review_hides_condition_and_keeps_mapping(work_tmp):
    root = Path(work_tmp)
    source = root / "pairs.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "condition", "question",
                                              "gold_answer", "pred_answer", "error"])
        writer.writeheader()
        writer.writerows([
            {"qid": "q1", "condition": "baseline", "question": "问题", "gold_answer": "标准",
             "pred_answer": "回答甲", "error": ""},
            {"qid": "q1", "condition": "remove_gold", "question": "问题", "gold_answer": "标准",
             "pred_answer": "回答乙", "error": ""},
        ])
    blind_path, reviewer_1_path, reviewer_2_path, key_path, _ = prepare(
        source, root / "review", seed=7)
    with blind_path.open(encoding="utf-8-sig", newline="") as f:
        blind = list(csv.DictReader(f))
    with key_path.open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    assert len(blind) == len(key) == 2
    assert "qid" not in blind[0] and "condition" not in blind[0]
    assert {r["review_id"] for r in blind} == {r["review_id"] for r in key}
    assert {r["model_answer"] for r in blind} == {"回答甲", "回答乙"}
    assert {r["condition"] for r in key} == {"baseline", "remove_gold"}
    assert reviewer_1_path.read_bytes() == blind_path.read_bytes()
    assert reviewer_2_path.read_bytes() == blind_path.read_bytes()
