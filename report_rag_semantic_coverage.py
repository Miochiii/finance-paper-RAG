"""Build the descriptive pilot report and a standalone scientific figure."""
import argparse
import csv
from pathlib import Path

from rag_fresh_change_experiment import load


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def report(output):
    summary = load(output / "analysis_summary.json")
    audit = load(output / "integrity_audit.json")
    scores, screen = rows(output / "answer_scores.csv"), rows(output / "screening_summary.csv")
    conditions = {"baseline": "原上下文", "semantic_partial": "语义缺失", "retained_evidence_prefix": "证据保留前缀"}
    lines = ["# 语义缺失与无标注覆盖信号试验结果", "",
        f"筛查 {summary['screened_questions']} 题，生成前合格 {summary['qualified_questions']} 题；生成 {summary['case_count']} 条新回答。完整回答定义为离线质量 2，与拒答规则解耦。", "",
        "## 生成前筛查", "", "|题号|组别|合格|删除尝试数|目标点|删除字符数|", "|---|---|---:|---:|---:|---:|"]
    for r in screen:
        lines.append(f"|{r['qid']}|{r['cohort']}|{r['eligible']}|{r['attempts']}|{r['target_index']}|{r['deleted_chars']}|")
    lines += ["", "每个合格语义缺失案例，两次核查都确认目标 absent、其他事实 supported。筛查失败在结果目录中完整保留，未根据新生成回答排除案例。", "",
              "未合格原因："]
    for r in screen:
        if r["eligible"] == "1":
            continue
        construction = load(output / "construction" / (r["qid"] + ".json"))
        if any(s != "supported" for run in construction["baseline_statuses"] for s in run):
            reason = "原上下文的两次自动支持核查未同时通过（包括摘录不逐字匹配时的 U）。"
        else:
            reason = "在预设删除次数内，目标仍有支持，或删除连带破坏其他必要事实的支持。"
        lines.append(f"- {r['qid']}：{reason}")
    lines += ["", "## 配对结果", "", "|组别|条件|题数|质量分布 0/1/2/U|规则拒答|问题—回答自查|补充检索覆盖|",
              "|---|---|---:|---|---:|---|---|"]
    for g in summary["groups"]:
        if g["n"]:
            lines.append(f"|{g['cohort']}|{conditions[g['condition']]}|{g['n']}|{g['qualities']}|{g['refusals']}|{g['direct']}|{g['retrieved']}|")
    lines += ["", "## 信号可用性与错误", "", "以质量 0/1 为不完整、质量 2 为完整；U 不填为正常。以下为案例级描述，不是变点序列检出率。", "",
              "|组别|信号|已知质量数|信号可用数|不完整数|检出|确定漏报|不完整 U|完整数|完整误报|完整 U|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in summary["method_metrics"]:
        if m["n"]:
            lines.append("|" + "|".join(str(m[k]) for k in ("cohort", "method", "n", "determinate", "incomplete",
              "flagged_incomplete", "missed_incomplete", "unresolved_incomplete", "complete", "flagged_complete", "unresolved_complete")) + "|")
    lines += ["", "## 逐题对照", "", "|题号|条件|质量 原→变体|拒答|自查 原→变体|补充检索 原→变体|相关性缺口变化 512/4096|",
              "|---|---|---|---:|---|---|---|"]
    for p in summary["pairs"]:
        lines.append(f"|{p['qid']}|{conditions[p['condition']]}|{p['baseline_quality']}→{p['variant_quality']}|{p['variant_refusal']}|"
           f"{p['baseline_direct']}→{p['variant_direct']}|{p['baseline_retrieval']}→{p['variant_retrieval']}|{p['delta_gap512']:.6f}/{p['delta_gap4096']:.6f}|")
    quiet = summary["non_refusal_quality_declines"]
    lines += ["", f"本轮质量下降且没有规则拒答的案例数：{len(quiet)}。对应题号及条件：" +
              "、".join(p["qid"] + "/" + p["condition"] for p in quiet) + "。", "",
              "## 解释与局限", "",
              "- 问题—回答自查只能判断明示要求有无回应。回答自信地补造完整但错误的事实时，它可能漏报。",
              "- 补充检索要点全部先于回答生成；代理请求不含冻结标准事实。使用知识库证据不代表完全无来源信息，当前也没有验证生产检索器。",
              "- 检索库来自原 40 题正常缓存上下文。缓存证据与原题有直接关系；本实验只能证明这种补证据机制在当前资料上的作用。",
              "- 开发组含上一轮已发现的失败题。来源核查组虽未参加上一轮隐蔽故障试验，仍属于原 40 题池。筛查无合格来源核查题时，推广效果没有得到验证。",
              "- DeepSeek 同时负责生成、要点提取与评价；两次核查不等于独立审阅，也不消除同模型偏差。必要性/完整性的语义判断仍是自动判断。",
              "- 整句删除可能连带删除其他事实，尤其多个参数同在一句或两个要点互相蕴含；这些失败属于干预构造限制，不能视为检测器成功或失败。",
              "- 新信号仍须独立正常流校准、弱变化检出试验、成本/延迟测量。此处的案例级检出和误报计数不能替代长期变点实验。", "",
              "## 下一步", "",
              "先将补充检索覆盖作为候选信号，核查临时要点与最低回答范围的偏差；针对单句多参数增加可追踪的局部删除构造，并在新来源建立有效配对案例。之后用新生成序列独立冻结正常校准参数，再做弱漂移与长正常流。若代理在完整答案上误报，先修正要点范围，不直接调报警阈值掩盖问题。", "",
              "## 执行与审计", "",
              f"共 {summary['api_requests']} 个 API 响应，累计 {summary['total_tokens']} token；实际模型：{summary['models']}。失败记录数：{summary['failed_api_records']}。",
              f"512 token 特征截断 {summary['feature_truncated512']}/{summary['case_count']}；4096 token 特征截断 {summary['feature_truncated4096']}/{summary['case_count']}。",
              f"审计：{audit['status']}；响应 ID {audit['unique_response_ids']} 个且唯一，代码/输入/模型哈希检查 {audit['code_input_model_hashes_checked']} 项；验证生成前阶段锁、代理字段白名单、筛查失败保留、所有配对答案保留。", "",
              "图中颜色仅用于显示各信号自己的量表；质量缺口=(2-quality)/2 只是展示换算，不能与代理分数当作同一测量量比较。U 为灰色。", "",
              f"![配对信号]({(output / 'semantic_coverage_signals.png').resolve().as_posix()})", ""]
    v2_path, v3_path = output / "coverage_refinement_v2", output / "coverage_refinement_v3"
    if (v2_path / "analysis_summary.json").exists() and (v3_path / "analysis_summary.json").exists():
        v2, v3 = load(v2_path / "analysis_summary.json"), load(v3_path / "analysis_summary.json")
        m2, m3 = v2["metrics"][0], v3["metrics"][0]
        lines += ["## 同批回答上的代理修订（探索性开发）", "",
            "保留首版后，修订最低必要集合与具体数值，附上检索原文；第二版曾把原文支持误当回答覆盖。第三版明确待评价对象为回答，要求回答中的逐字摘录，并使用两次分类一致的结果。短摘录仅扩展为原文中的完整相邻句段，事实内容与自动语义核查意见不改写。", "",
            "|版本|可用数/12|不完整检出/4|确定漏报|不完整 U|完整误报/8|完整 U|",
            "|---|---:|---:|---:|---:|---:|---:|",
            "|首版|7|2|0|2|2|3|",
            f"|第二版|{m2['determinate']}|{m2['flagged_incomplete']}|{m2['missed_incomplete']}|{m2['unresolved_incomplete']}|{m2['flagged_complete']}|{m2['unresolved_complete']}|",
            f"|第三版|{m3['determinate']}|{m3['flagged_incomplete']}|{m3['missed_incomplete']}|{m3['unresolved_incomplete']}|{m3['flagged_complete']}|{m3['unresolved_complete']}|", "",
            "**这些版本使用同一批 12 条回答，后续版本是在查看前版结果后开发的，不是新的独立测试。第三版结果说明修复机制在开发案例上有效，不能推出总体检出率、推广性能或长期误报率。**", "",
            f"第二版新增 {v2['api_requests']} 次响应、{v2['total_tokens']} token；第三版新增 {v3['new_api_responses']} 次响应、{v3['total_tokens']} token。三阶段总响应数 {summary['api_requests']+v2['api_requests']+v3['new_api_responses']}，总 token {summary['total_tokens']+v2['total_tokens']+v3['total_tokens']}。没有新增 12 条生成回答，仍是初次生成的 12 条。", "",
            "当前可以冻结第三版作为下一次验证的候选。先解决新增来源的有效干预构造，完成独立于此次方法修订的新生成配对验证；然后才做长正常流与弱漂移校准。", ""]
    (output / "semantic_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")
    figure(output, scores)


def figure(output, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    order = {"baseline": 0, "semantic_partial": 1, "retained_evidence_prefix": 2}
    scores = sorted(scores, key=lambda r: (r["qid"], order[r["condition"]]))
    v2path, v3path = output / "coverage_refinement_v2/refined_answer_scores.csv", output / "coverage_refinement_v3/span_repaired_answer_scores.csv"
    v2 = {r["aid"]: r for r in rows(v2path)} if v2path.exists() else {}
    v3 = {r["aid"]: r for r in rows(v3path)} if v3path.exists() else {}
    columns = ["Quality deficit", "Refusal rule", "Question / answer", "Retrieval v1"]
    if v2 and v3:
        columns += ["Retrieval v2", "Retrieval v3"]
    columns += ["BGE gap 512", "BGE gap 4096"]
    matrix = []
    for r in scores:
        values = [(2-int(r["quality"]))/2 if r["quality"] != "U" else np.nan, float(r["rule_refusal"])]
        values.extend(float(r[k]) if r[k] else np.nan for k in ("direct_score", "retrieval_score"))
        if v2 and v3:
            values += [float(v2[r['aid']]['refined_score']) if v2[r['aid']]['refined_score'] else np.nan,
                       float(v3[r['aid']]['v3_score']) if v3[r['aid']]['v3_score'] else np.nan]
        values.extend(float(r[k]) for k in ("gap512", "gap4096"))
        matrix.append(values)
    matrix = np.asarray(matrix)
    labels = {"baseline": "original", "semantic_partial": "one fact absent", "retained_evidence_prefix": "retained + prefix"}
    fig, ax = plt.subplots(figsize=(14, max(5.5, len(scores)*.43)))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#dedede")
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=9)
    ax.set_yticks(range(len(scores)), [r["qid"] + "  " + labels[r["condition"]] for r in scores], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "U" if np.isnan(value) else f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if not np.isnan(value) and value > .65 else "#222")
    for i in range(3, len(scores), 3):
        ax.axhline(i-.5, color="#777", lw=1)
    ax.set_title("Semantic absence pilot: paired outcomes and annotation-free proxies", fontsize=12, pad=18)
    fig.text(.5, .02, "Higher values denote a larger deficit within each column. Gray = unresolved. Revisions reuse these development answers; descriptive only.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "semantic_coverage_signals.png", dpi=190)
    fig.savefig(output / "semantic_coverage_signals.svg")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    report(ap.parse_args().output)
