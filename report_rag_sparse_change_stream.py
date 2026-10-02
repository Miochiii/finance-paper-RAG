"""Report all prespecified V10 sparse-stream detector configurations."""
import argparse
import csv
import os
from pathlib import Path
from rag_fresh_change_experiment import load


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def report(output, log):
    m=load(output/"protocol_lock.json");cfg=m["config"];s=load(output/"analysis_summary.json")
    audit=load(output/"integrity_audit.json");split=load(output/"source_split.json")
    lines=["# V10：低频间歇数值证据故障的新生成序列","","## 实验范围","",
        f"本轮共{s['fresh_generations']}条新生成回答，响应ID全部唯一；均值校准{cfg['mean_n']}条，阈值校准{cfg['calibration_runs']}条流，正常测试{cfg['normal_runs']}条流，三档故障测试各{cfg['fault_runs_per_rate']}条流。",
        f"监控流长度{cfg['horizon']}，第{cfg['change_at']}次之后，各请求以10%、30%、60%的预设概率替换为冻结的数值故障证据；随机计划先于任何生成锁定。故障概率是注入概率，不是直接假定的实际错误率。",
        "沿用V9通过干预检查的10题/8篇已有论文与固定检索缓存。4篇用于校准，另外4篇用于测试，每组5题。论文在之前开发中已见过；只是在本轮按来源分开，不属于完全未见论文验证。",
        "题目按均匀选论文、再均匀选该论文题目抽取。每个位置重新调用生成API；每条流内生成、评分处理串行，不按报警停止采集；多个流并行，正常与不同故障档流的启动顺序按种子打散。",
        "完全相同的问题、回答、完成原因和两份正常核对事实才共享本轮评分缓存；缓存首次建立需两轮比较并核验回答引文。没有复用旧生成答案或旧阶段评分。本规则在新数据之前冻结。",
        f"两轮评分的不同输入{s['distinct_score_inputs']}组、评分API响应{s['scoring_api_responses']}次、缓存命中{s['score_cache_hits']}次。缓存共享使评分观测相关；不能把生成条数称为同等数量的独立评审。",
        "来源、参考、生成/监测载荷和评分仍是MinerU原文与同供应商自动评价；正常监测依据未被故障污染。缓存检索的受控序列不是自然线上流量。","",
        "## 信号、方法与校准","",
        "主信号：正常证据核对incomplete记1、complete记0；U按预设规则记1并单独报告。拒答为独立基线信号，不把未定丢弃或当正确。",
        f"核对信号均值上界m={s['calibration']['m']:.6f}，仅由新均值校准得到二项上界(delta={cfg['delta']})；CUSUM的p1={s['calibration']['cusum_p1']:.3f}（规则预设为.3，若m≥.3则取(m+1)/2）。",
        "提前指定8项方法：首次拒答、首次核对异常、5条滑窗至少2异常、滑窗正常校准阈值、Bernoulli CUSUM固定/校准阈值、7λ均匀混合e-CUSUM固定100/校准阈值。没有按测试挑选最优配置。",
        "校准阈值使用12条正常校准流最大统计量的预设秩；目标视界误报10%，测试从不参与阈值或m估计。论文组不同、同题重复及服务依赖意味着不能将该秩规则直接宣称为未来误报保证。",
        "e-detector文献提供的是满足模型前提时的ARL控制，阈值100不能解释为40次请求内误报率1%；本项目用样本估计的m也未证明未来条件均值约束。[原论文](https://nejsds.nestat.org/journal/NEJSDS/article/59/text)","",
        "## 回答质量与信号","","|阶段|证据条件|N|完整|不完整|质量U|核对U|异常信号|不完整检出|错误拒答|完整误报|","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["metrics"]:
        lines.append("|"+"|".join(str(r[k]) for k in ("phase","condition","n","complete","bad","quality_u","proxy_u","positives","bad_flagged","bad_refusals","complete_false_flags"))+"|")
    lines += ["","## 全部预设检测结果","",
        "正常报警按40次内至少一次首次报警统计；提前报警不算变后检出。检出者平均延迟排除漏检；受限延迟在无提前报警流计算，漏检记为25次，不是无限视界EDD或ARL。",
        "各档包含所有6条预定故障流，实际未注入任何故障的流也保留。三档复用同一批30条正常测试，仅不同方法重放，正常流不重复计成90条。","",
        "|故障概率|方法|阈值|正常报警/30|提前报警/6|检出/6|检出者平均延迟|受限延迟|","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in s["detector_results"]:
        delay="—" if r['mean_detected_delay'] is None else f"{r['mean_detected_delay']:.2f}"
        restricted="—" if r['restricted_post_delay'] is None else f"{r['restricted_post_delay']:.2f}"
        lines.append(f"|{r['rate']:.0%}|{r['method']}|{r['threshold']:.5g}|{r['false_alarms']}|{r['pre_alarms']}|{r['detected']}|{delay}|{restricted}|")
    r=s['detector_results'][0]
    lines += ["",f"各方法完整精确二项区间见detector_results.csv；例如首行正常视界误报95%区间[{r['far_ci_low']:.4f}, {r['far_ci_high']:.4f}]。这些区间仅在流间IID假设下描述抽样误差；未涵盖题库/论文相关性、缓存共享和云服务时变，不当作实际保证。", "",
        "## 实际故障位置","","|流|故障概率|注入数|变后不完整|变后质量U|变后核对异常|","|---|---:|---:|---:|---:|---:|"]
    for r in read_csv(output/"stream_realizations.csv"):
        if float(r['rate'])>0:
            lines.append("|"+"|".join(str(r[k]) for k in ("stream","rate","injected","post_bad","post_u","post_coverage_flags"))+"|")
    lines += ["","## 校准与测试来源",""]
    for group,sources in split.items():
        lines.append(f"### {group}\n")
        lines += [f"- {source}：{', '.join(qids)}" for source,qids in sources.items()]
        lines.append("")
    lines += ["## 解释与后续","",
        "若正常信号仍全零，首次核对异常往往最快；同样观测到0报警并不等于各方法真实误报预算相同。校准型累计方法也可能退化为首次正信号就报警，不能声称e-detector胜出。",
        "固定阈值方法若在10%故障下漏检，要如实报告稀疏故障的检出/延迟代价，不用测试调低阈值或改变m补分。后续若需调整，另立开发阶段与新的验证数据。",
        "30条正常流即使0报警，双侧95%二项区间上限仍约11.6%，无法确认5%误报目标；不能把重复题目、更多API响应ID当作独立来源样本。",
        "下一步依据全部结果决定是否需收集更复杂的正常请求、无害证据变化及自然弱故障。在比较统计器优势之前，先补充能检验真实误报代价的正常分布。无需新增人工复核，自动评价限定持续保留。",
        f"实际响应模型{s['actual_models']}，生成结束{s['generation_finishes']}；总tokens {s['total_tokens']}；失败评分记录{s['failed_api_records']}。",
        f"完整性与时间顺序审计：{audit['status']}。",""]
    (output/"sparse_stream_report.md").write_text("\n".join(lines),encoding="utf-8")
    log.mkdir(parents=True,exist_ok=True)
    (log/"稀疏故障新生成序列V10结果_20261002.md").write_text("\n".join(lines),encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR",str((output/"mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=["first_refusal","first_coverage","window5_two","cusum_fixed","edetector_fixed100"]
    labels=["First refusal","First coverage","Window 5: >=2","CUSUM fixed","E-detector 100"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for method,label in zip(methods,labels):
        rr=[r for r in s['detector_results'] if r['method']==method]
        x=[r['rate']*100 for r in rr]
        axes[0].plot(x,[r['detected']/r['n_fault'] for r in rr],marker='o',label=label)
        axes[1].plot(x,[r['restricted_post_delay'] for r in rr],marker='o',label=label)
    axes[0].set(title="Detection within 24 post-change queries",ylabel="Detected fraction",ylim=(-.04,1.04))
    axes[1].set(title="Restricted delay (misses = 25)",ylabel="Queries",ylim=(0,26))
    for ax in axes:
        ax.set(xlabel="Post-change evidence fault probability (%)",xticks=[10,30,60])
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9,loc='best')
    fig.suptitle(f"V10: {s['fresh_generations']} fresh answers; 30 normal + 18 sparse-fault test streams",fontsize=12)
    fig.supxlabel("Fixed methods shown; all 8 prespecified methods in report. Conditional automatic truth, existing source bank.",fontsize=9)
    for ext in ("png","svg"):fig.savefig(output/("sparse_stream_results."+ext),dpi=180)
    plt.close(fig)
    print("Saved",output/"sparse_stream_report.md")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log)
