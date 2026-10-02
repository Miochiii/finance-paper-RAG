"""Descriptive report of frozen coverage validation; no test-result tuning."""
import argparse
import csv
from pathlib import Path
from rag_fresh_change_experiment import load


def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))


def report(output):
    summary,audit=load(output/"analysis_summary.json"),load(output/"integrity_audit.json")
    screen,answers,pairs=[read_rows(output/n) for n in ("screening_summary.csv","validation_answer_scores.csv","validation_paired_differences.csv")]
    names={"baseline":"原上下文","semantic_partial":"局部语义缺失","retained_evidence_prefix":"证据保留前缀","all":"全部"}
    lines=["# 冻结第三版覆盖代理：新增来源与新生成回答验证", "",
      f"筛查 {summary['screened_questions']} 篇不同文献，{summary['paired_questions']} 题得到有效局部干预，生成 {summary['generated_answers']} 条新配对回答。代理提示词、范围提取和判定函数与第三版相同，在生成前冻结所有临时要点。", "",
      "来源未参加之前的隐蔽故障生成或代理修订；仍来自原 40 题池，参加过早期原文评价与强故障试验。不能称为完全未用过的外部测试集。", "",
      "## 完整筛查记录", "", "|题号|原上下文支持完整|干预合格|进入配对验证|目标事实|尝试数|", "|---|---:|---:|---:|---:|---:|"]
    for r in screen:lines.append("|"+"|".join(r[k] for k in ("qid","baseline_supported","eligible","selected","target_index","attempts"))+"|")
    unsupported=sum(r['baseline_supported']=='0' for r in screen)
    coupled=sum(r['baseline_supported']=='1' and r['eligible']=='0' for r in screen)
    lines += ["",f"{unsupported} 题原缓存上下文不能完整支持全部冻结必要事实，未进入删除试验；{coupled} 题基线支持完整，但在预设次数内未能只移除目标事实。失败保留，没有依新生成答案质量重新选题。", "",
      "局部片段删除记录每轮输入、真实原文偏移、删除文字和输出。参数题 fin_013 的有效干预移除了过采样前 n_estimators=100 的支持，保留其他三个参数；这是人工构造的证据缺失，尚不是自然线上故障。", "",
      "## 固定方法的新回答结果", "", "|条件|答案数|质量 U|完整|不完整|代理 U|不完整检出|确定漏报|不完整 U|完整误报|规则拒答|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in summary['metrics']:
        lines.append("|"+names[m['condition']]+"|"+"|".join(str(m[k]) for k in ('n','quality_u','complete','incomplete','signal_u','flagged_incomplete','missed_incomplete','unresolved_incomplete','flagged_complete','refusals'))+"|")
    lines += ["", "## 逐题配对", "", "|题号|条件|质量 原→变体|代理 原→变体|变体拒答|相关性缺口变化 512/4096|", "|---|---|---|---|---:|---|"]
    for r in pairs:lines.append(f"|{r['qid']}|{names[r['condition']]}|{r['baseline_quality']}→{r['variant_quality']}|{r['baseline_signal']}→{r['variant_signal']}|{r['variant_refusal']}|{float(r['delta_gap512']):.6f}/{float(r['delta_gap4096']):.6f}|")
    gate=summary['gate']
    lines += ["", "## 预设机制门槛", "", f"结果：**{gate['status']}**。", "", "|检查|结果|", "|---|---|"]
    for key,passed in gate['checks'].items():lines.append(f"|{key}|{'通过' if passed else '未通过'}|")
    lines += ["", "本轮没有改提示词、改判定规则、删掉误报或调整门槛。门槛在筛查和生成前设定，失败原因全部保留。即使某些案例全部检出，样本量、U 或完整答案误报未满足条件时，仍不能宣布门槛通过。", "",
      "## 解释与下一步", "",
      "- 补充检索取证来自原题正常证据缓存，当前验证没有评估完整 MinerU 知识库的生产检索。",
      "- 普通缓存中缺少必要事实时，拒答率为零也不能证明回答完整；先检查正常检索的事实覆盖，再用代表性的正常序列校准。",
      "- 有效干预题只占筛查的一部分，这会选择较容易隔离事实且基线完整的题；不能推广到全部问题。",
      "- DeepSeek 同时生成、提取和评价；两次核查不等于独立人工标注，完整/不完整是冻结事实下的自动判断。",
      "- 本轮案例级识别不是变点检出率，不能据此推断长正常流误报或弱变化的检出延迟。", ""]
    if gate['status']=='passed':
        lines += ["下一步可以制定并冻结长正常流、弱变化流与简单检测基线的校准方案，再按该方案重新调用 API。", ""]
    else:
        lines += ["下一步先对筛查失败的来源进行证据覆盖诊断：比较原缓存和 MinerU 原文检索，确认是否是普通上下文缺失、事实范围不一致或要点耦合。补充验证题池并预先固定抽样与干预规则；对冻结代理发现的误报保留为验证失败，后续修订需作为新开发版本重新验证。当前不进入长序列性能宣称。", ""]
    lines += ["## 执行与审计", "",
      f"API 响应 {summary['api_responses']} 个，累计 {summary['total_tokens']} token，实际模型 {summary['models']}；失败 API 记录 {summary['failed_api_records']}。",
      f"特征截断：512 token {summary['feature_truncated512']}/{summary['generated_answers']}，4096 token {summary['feature_truncated4096']}/{summary['generated_answers']}。",
      f"完整审计 {audit['status']}：新响应 ID {audit['unique_new_api_response_ids']} 个唯一且未复用开发响应；{audit['code_input_model_hashes_checked']} 项代码、输入、模型哈希；核查阶段锁、准确原文删除、代理请求白名单、提示词不变和所有派生分数。", "",
      f"![验证信号图]({(output/'validation_signals.png').resolve().as_posix()})", ""]
    if (output/'failure_cause_summary.json').exists():
        failure=load(output/'failure_cause_summary.json')
        lines += ['## U 的原因诊断（不改变原结果）','',
          f"代理原因计数：{failure['signal_cause_counts']}。质量 U {failure['quality_u']} 条，其中 {failure['quality_u_with_inconsistent_total']} 条至少一次出现总质量为2、分项却为partial/missing的自相矛盾。程序保留U，没有自行猜分。",'',
          '回答摘录校验失败包括：只引用AUC等短实体不足4字、省略回答中的Markdown加粗符号、以及对原句作了改写。真实语义回应存在时也可能因此被严格校验置为U。此处仅解释原因，不把这些结果事后改判为正确。','',
          '另外，fin_007 的冻结事实包含多期动态观点更新，而问题问模型是什么；临时最低定义集合并未要求这一扩展。fin_022 的冻结集合包含ROC与AUC，而部分缓存/自动提取集合只列AUC。必要范围的差异需单独核查，不能只根据已生成回答调整标准。','']
    diagnostic=output/'mineru_retrieval_diagnostic/analysis_summary.json'
    if diagnostic.exists():
        d=load(diagnostic)
        lines += ['## MinerU 本地证据诊断','',
          f"不调用API、不把标准事实用于检索。对12题40个冻结摘录，本地问题检索命中 {d['retrieved_quote_matches']}/40，原缓存 {d['cache_quote_matches']}/40；实际可见来源全文包含 {d['whole_visible_source_quote_matches']}/40。{d['improved_questions']}题逐字命中增加，{d['worsened_questions']}题减少。",'',
          '**逐字命中仅说明相同原文可获得，不能证明语义覆盖、回答正确或新检索优于原检索。** 这说明可以继续从现有MinerU原文改进取证，尚不需要重新解析PDF。','']
    (output/'validation_report.md').write_text('\n'.join(lines),encoding='utf-8')
    figure(output,answers)


def figure(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names={'baseline':'original','semantic_partial':'one fact absent','retained_evidence_prefix':'retained + prefix'}
    order={'baseline':0,'semantic_partial':1,'retained_evidence_prefix':2}
    rows=sorted(rows,key=lambda r:(r['qid'],order[r['condition']]))
    values=[]
    for r in rows:values.append([(2-int(r['quality']))/2 if r['quality']!='U' else np.nan,float(r['rule_refusal']),
       float(r['signal_score']) if r['signal_score'] else np.nan,float(r['gap512']),float(r['gap4096'])])
    values=np.asarray(values);cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#ddd')
    fig,ax=plt.subplots(figsize=(10.8,max(5,len(rows)*.46)))
    ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(5),['Quality deficit','Refusal rule','Frozen coverage v3','BGE gap 512','BGE gap 4096'],fontsize=9)
    ax.set_yticks(range(len(rows)),[r['qid']+'  '+names[r['condition']] for r in rows],fontsize=9)
    for i in range(len(rows)):
        for j in range(5):
            v=values[i,j];ax.text(j,i,'U' if np.isnan(v) else f'{v:.3f}',ha='center',va='center',fontsize=9,color='white' if not np.isnan(v) and v>.65 else '#222')
    for i in range(3,len(rows),3):ax.axhline(i-.5,color='#777',lw=1)
    ax.set_title('Frozen coverage proxy: new responses on non-development sources',fontsize=12,pad=16)
    fig.text(.5,.02,'Descriptive selected-case results. Higher = larger deficit within each column. Gray = unresolved.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(output/'validation_signals.png',dpi=190);fig.savefig(output/'validation_signals.svg');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    report(ap.parse_args().output)
