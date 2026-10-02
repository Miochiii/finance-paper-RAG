import json

from export_online_quality import parse_events


def test_export_filters_old_and_failed_events_and_sorts_by_time():
    data = [
        {"event": "ask", "t": 4.0, "rid": "later", "origin": "mcp",
         "quality_proxy_version": "explicit_v1", "ok": True,
         "ans_refusal_explicit": 1, "question_hash": "a"},
        {"event": "ask", "t": 2.0, "rid": "old", "origin": "mcp", "ok": True},
        {"event": "ask", "t": 3.0, "rid": "error", "origin": "http",
         "quality_proxy_version": "explicit_v1", "ok": False},
        {"event": "ask", "t": 1.0, "rid": "earlier", "origin": "http",
         "quality_proxy_version": "explicit_v1", "ok": True,
         "ans_refusal_explicit": 0, "question_hash": "b"},
    ]
    rows, counts = parse_events([json.dumps(x) for x in data])
    assert [r["rid"] for r in rows] == ["earlier", "later"]
    assert counts["all_asks"] == 4
    assert counts["legacy_no_proxy"] == 1
    assert counts["failed_or_missing"] == 1
