"""Post-result semantic validity review; never rewrite the primary experiment.

Existing blind extractions were from frozen retrieved evidence, not a new
pre-answer extraction from source cards. The new guard/unit rule is development
only. No fresh scoring calls, outcome exclusions or claimed prospective gain.
"""
import argparse
import json
from pathlib import Path

from rag_exact_source_validation import verify
from rag_fresh_change_experiment import load, dump, digest, utc, write_csv
from rag_objective_slots import checks_pass
from rag_objective_admission_guard import source_reference_consistency


def review(output, log):
    verify(output)
    primary = load(output / "analysis_summary.json")
    audit = load(output / "integrity_audit.json")
    verification_path = output / "verification_results.json"
    verification = load(verification_path) if verification_path.exists() else None
    questions = load(output / "public_questions.json")
    references = load(output / "private_references.json")
    keys = load(output / "private_question_key.json")
    inputs = [output / name for name in (
        "analysis_summary.json", "public_questions.json", "private_references.json",
        "private_question_key.json", "public_cases.json", "private_case_key.json",
        "private_control_answers.json", "integrity_audit.json", "protocol_lock.json",
        "questions_lock.json", "source_integrity_lock.json", "retrieval_lock.json",
        "proxy_lock.json", "cases_lock.json", "validation_tools_lock.json",
        "public_retrieval.json", "prospective_objective_report.md",
    )]
    for name in ("proxy_plans", "scores", "generations", "candidates", "api_records"):
        inputs.extend(sorted((output / name).glob("*.json")))
    if verification:inputs.append(verification_path)
    hashes = {str(p.resolve()): digest(p) for p in inputs}
    destination = output / "validity_review"
    destination.mkdir(exist_ok=True)
    rows = []
    traces = {}
    redundant_percent_fields = []
    for qid, question in questions.items():
        original = load(output / "proxy_plans" / (qid + ".json"))
        reference = references[qid]
        candidate = dict(question=question["question"], slots=reference["slots"])
        result = source_reference_consistency(candidate, original["raw"], original["checks"], original["evidence"])
        admission = load(output / "candidates" / (keys[qid]["input_id"] + ".json"))
        gold_checks = len(admission["checks"]) == 2 and all(checks_pass(c, len(candidate["slots"]),
            ("explicit_required", "expected_supported", "unique", "aliases_valid"),
            ("question_clear", "scope_complete")) for c in admission["checks"])
        conflicts = [r for r in result["reasons"] if r["reason"] == "price_field_reference_uses_time_unit"]
        redundant = [s["index"] for s in reference["slots"]
                     if s["type"] == "number" and s["expected"].endswith(("%", "％"))
                     and s["unit"] in ("%", "％")]
        redundant_percent_fields.extend(dict(question_key=qid, index=i) for i in redundant)
        row = dict(question_key=qid, source=keys[qid]["source"],
                   primary_basis_complete=int(original["basis_complete"]),
                   development_basis_complete=int(result["plan"]["basis_complete"]),
                   development_consistency_pass=int(result["admit"]),
                   gold_aware_admission_checks_pass=int(gold_checks),
                   field_unit_conflict_fields=len(conflicts),
                   numeric_unit_mapped_fields=len(result["plan"]["numeric_unit_mappings"]),
                   duplicated_percent_unit_fields=len(redundant),
                   reasons="|".join(sorted({r["reason"] for r in result["reasons"]})))
        rows.append(row)
        traces[qid] = dict(question=question, reference=reference,
                           original_proxy_basis=original["basis_complete"],
                           candidate_source_checks=admission["checks"],
                           development_consistency=result)
    for filename, expected in hashes.items():
        if digest(Path(filename)) != expected:
            raise ValueError("primary experiment changed during review: " + filename)
    write_csv(destination / "validity_diagnostic.csv", rows)
    dump(destination / "private_validity_trace.json", traces)
    metrics = {r["condition"]: r for r in primary["metrics"]}
    live, controls = metrics["live_normal"], metrics["controls"]
    summary = dict(
        reviewed_at=utc(), scope="post_result_development_and_reference_validity_review",
        primary_numeric_gate=primary["gate"]["status"],
        overall_semantic_validity="needs_revision_before_confirmatory_use",
        primary_metrics_unchanged=True, no_questions_excluded=True,
        primary_questions=primary["qualified_questions"], primary_sources=primary["qualified_sources"],
        primary_automatic_reference_complete=live["complete"],
        primary_automatic_reference_complete_is_not_confirmed_real_accuracy=True,
        primary_proxy_bases=primary["proxy_bases"],
        primary_bad_controls_flagged=controls["flagged_incomplete"],
        primary_bad_controls_unresolved=controls["unresolved_incomplete"],
        primary_complete_controls_false_flags=metrics["control_complete"]["false_flags"],
        reference_field_unit_conflict_questions=sum(r["field_unit_conflict_fields"] > 0 for r in rows),
        numeric_unit_representation_repaired_questions=sum(
            not r["primary_basis_complete"] and r["development_basis_complete"] for r in rows),
        redundant_percent_representation_questions=len({r["question_key"] for r in redundant_percent_fields}),
        redundant_percent_representation_fields=len(redundant_percent_fields),
        development_basis_complete=sum(r["development_basis_complete"] for r in rows),
        development_consistency_pass=sum(r["development_consistency_pass"] for r in rows),
        development_replay_not_new_scoring_or_prospective_validation=True,
        blind_replay_used_retrieved_evidence_not_new_source_card_requests=True,
        new_api_calls=0, historical_dataflow_integrity_audit=audit["status"],
        historical_audit_does_not_certify_reference_semantics=True,
        future_admission_requires_blind_source_cards_before_test_answers=True,
        independent_human_truth=False,
        targeted_tests_passed=verification["tests"]["passed"] if verification else None,
    )
    dump(destination / "validity_summary.json", summary)
    dump(destination / "review_inputs_lock.json", dict(created_at=utc(),
         after_primary_results=True, hashes=hashes, code_hashes={str(p.resolve()): digest(p) for p in (
             Path(__file__), Path(__file__).with_name("rag_numeric_unit_mapping.py"),
             Path(__file__).with_name("rag_objective_admission_guard.py"),
             Path(__file__).parent / "tests/test_rag_numeric_unit_mapping.py",
             Path(__file__).parent / "tests/test_rag_objective_admission_guard.py")},
         no_reference_or_primary_score_edits=True))
    lines = [
        "# 前瞻新题验证V7.1：有效性复查及开发修复", "",
        "## 当前结论", "",
        "**预设数值门槛通过，但参考答案有效性仍需修订。本轮不能宣称16题全部真实正确，也不能宣称变点检测已完成验证。**", "",
        f"本轮8篇论文、16道新题；新生成16条正常回答，另有48条合成对照。自动参考评分为完整{live['complete']}/16；遗漏/改错对照检出{controls['flagged_incomplete']}/32，另有{controls['unresolved_incomplete']}条未定。完整对照误报{metrics['control_complete']['false_flags']}/16，其中2条代理未定，不能将未定理解为正确判定。",
        "原始题库、标准答案、64例评分、检索配置和门槛全部保留。后续复查没有剔除题目，没有改写原成绩。", "",
        "## 发现的问题", "",
        "|项目|原始现象|判断与处理|", "|---|---|---|",
        "|P0001：题干/参考语义|字段要求Y_pre1等变量的“交易日后价格”，参考却填交易日数；盲提取与两轮核查均认为原文没有所问具体价格|来源表给的是预测间隔和二分类含义。先前两轮带参考的候选检查通过，显示自动评审可以共同接受错误字段关系。当前题需要修改问法并重新入选，不能靠参考去填补盲提取|",
        "|P0002：数值与单位|原文已经检到；临时提取把数值和单位一起放入expected，现行数值校验拒绝|新的开发规则只对有逐字来源的单个数字＋已知单位拆分，保留原文位置；没有单位换算或数值改动|",
        "|P0003：百分号表示|一个参考字段expected已含%，unit又为%；合成完整答案重复百分号|后续数值比较保留百分数的量级，构造答案只输出一次符号。不能把5%等同于无单位0.05；本轮原始对照保留|", "",
        "上述问题首先属于题目/评价管线；P0001和P0002的相关原文均已检索到，不能统一归因于检索漏检。", "",
        "## 离线开发回放", "",
        f"- 原规则核对依据有效{primary['proxy_bases']}/16；新单位规则离线回放有效{summary['development_basis_complete']}/16。",
        f"- 新准入一致性检查在现有盲提取上通过{summary['development_consistency_pass']}/16，P0001被拦下。",
        "- 回放复用历史检索原文和模型输出，没有新增回答或评分请求。它用于验证修复方向，不构成新的前瞻验证，也不产生更新的故障检出率。",
        "- 下一轮要在候选来源卡片上重新做不含标准答案的提取与两轮核查，然后才和拟定参考比较，全部流程先于待测回答。当前回放使用检索证据，不能冒充已执行这一步。", "",
        "## 原始实验的范围", "",
        "8篇论文未参与当时覆盖方法的开发/筛选，但早已存在于知识库和旧标注集；每篇2题，题目与合成对照不完全独立。当前复查后这些题和来源已成为开发资料，不能再次称为未见来源验证。",
        "候选、提取、两轮检查和评分使用同一服务。实际返回模型为deepseek-flash，请求模型为deepseek-chat。即便多轮一致，也不是独立人工真值。来源以MinerU原始table/text块为条件，未独立逐页验证PDF。",
        f"全程{primary['api_responses']}个新API响应，{primary['total_tokens']} tokens。完整性审计为{audit['status']}，核查的是请求、哈希、来源位置和成绩推导；它不能证明参考答案语义正确。",
        f"来源选择、新题构造、字段检索、句子边界、数值单位与准入检查的关联测试通过{summary['targeted_tests_passed']}项；其中新增修复测试17项。两幅结果图已检查布局与数字。",
        "测试回答生成前还纠正过一次来源卡片：旧准备把chart重建当作table，4道候选受影响。旧记录完整保留且未生成待测回答；新准备只选择原始table/text块并拒绝明确近似值。", "",
        "## 后续顺序", "",
        "1. 将新增的盲提取准入、一致性拒绝、严格单位拆分和百分号构造规则冻结到下一轮。P0001重新构造明确询问预测间隔的新候选，当前题保留在问题清单。",
        "2. 从未用于这些实验的原始table/text片段生成新题；先去重并记录全部候选/未定，再做来源盲提取、两轮检查和参考一致性核验。当前8篇已成为开发来源，若仍从现有38篇语料出题，应明确报告“新题、已有来源”；未见论文的结论需要另增论文。",
        "3. 预先冻结题库、方法、评价门槛和全部案例，再生成新的正常回答与单字段合成对照。通过有效性检查后，建立真实正常/证据故障配对；评分依据来自正常证据，干预仅作用于生成输入。",
        "4. 真实故障确实产生质量下降后再进入长序列变点实验，按论文/题目分组报告，并与简单拒答/滑窗基线比较。", "",
        "上述工作可以继续自动进行，无需新增人工复核；结论应始终写作来源条件下的自动评价。当前不修改正式RAG运行路径，也未提交或推送GitHub。", "",
        "## 文件", "",
        f"[原始数值报告]({(output/'prospective_objective_report.md').resolve().as_posix()})；[完整性审计]({(output/'integrity_audit.json').resolve().as_posix()})；[有效性汇总]({(destination/'validity_summary.json').resolve().as_posix()})。", "",
        "下图展示原始冻结成绩。中间的完整数以自动参考为条件，不能解读为独立确认的正确率。", "",
        f"![原始成绩与有效性限制]({(destination/'qualified_primary_results.png').resolve().as_posix()})", "",
    ]
    import os
    os.environ.setdefault("MPLCONFIGDIR", str((output / "mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    values = ([16, primary["proxy_bases"]], [live["complete"], live["quality_u"]],
              [controls["flagged_incomplete"], controls["unresolved_incomplete"]])
    labels = (["New questions", "Available bases"], ["Auto-reference complete", "Score unresolved"],
              ["Flagged", "Unresolved"])
    titles = ["Frozen primary evidence", "Conditional automatic scoring", "32 synthetic bad controls"]
    for ax, counts, names, title in zip(axes, values, labels, titles):
        ax.bar(range(len(counts)), counts, color=["#277da1", "#9b9b9b"])
        ax.set_xticks(range(len(counts)), names, fontsize=9)
        ax.set(title=title, ylim=(0, max(counts) + 3), ylabel="Count")
        ax.spines[["right", "top"]].set_visible(False)
        for i, count in enumerate(counts):ax.text(i, count + .3, str(count), ha="center")
    fig.suptitle("V7.1: numeric gate passed; reference validity needs revision", fontsize=14)
    fig.supxlabel("Source-conditioned automatic scores; not independent truth. Complete-control false flags: 0/16 (2 unresolved).", fontsize=9)
    for ext in ("png", "svg"):fig.savefig(destination / ("qualified_primary_results." + ext), dpi=180)
    plt.close(fig)
    content = "\n".join(lines)
    (destination / "prospective_validity_report.md").write_text(content, encoding="utf-8")
    log.mkdir(parents=True, exist_ok=True)
    (log / "前瞻新题验证V7_1有效性复查_20261002.md").write_text(content, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    review(args.output, args.log)
