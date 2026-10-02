"""Aggregate V11 source-preserving normal-context stress report and figure."""
import argparse
from pathlib import Path
from rag_fresh_change_experiment import load


def report(output):
    s = load(output/'analysis_summary.json')
    if load(output/'integrity_audit.json')['status']!='passed':
        raise ValueError('integrity audit required before report')
    labels = {'live_normal':'原始证据','live_reverse':'顺序反转','live_dedup':'重叠去重','live_distractor':'相关干扰在前'}
    lines = ['# V11 正常证据压力实验','',
      f"覆盖 {s['input_questions']} 道既有题、{s['input_sources']} 篇既有论文，生成 {s['fresh_generations']} 条新回答。",
      f"{s['excluded_before_answers']} 个条件案例在回答生成前未通过准入；全部保留筛选记录。",'',
      '## 结果','', '| 场景 | 新回答 | 完整 | 错误或遗漏 | 质量 U | 错误被检出 | 完整回答误报 | 核对 U |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['metrics']:
        lines.append(f"| {labels[r['condition']]} | {r['n']} | {r['complete']} | {r['bad']} | {r['quality_u']} | {r['bad_flagged']} | {r['complete_false_flags']} | {r['proxy_u']} |")
    bad = sum(r['bad'] for r in s['metrics'])
    false = sum(r['complete_false_flags'] for r in s['metrics'])
    unknown = sum(r['quality_u']+r['proxy_u'] for r in s['metrics'])
    lines += ['',f"相对同题新基线：变差 {s['paired_worsened']}，改善 {s['paired_improved']}，比较 U {s['paired_quality_u']}。",'',
      '## 解释','',
      '正常证据场景中的错误回答是生成质量错误；只有标准事实判定完整而核对信号判为不完整，才计为误报。',
      '本轮没有改变检测器、阈值、生成提示词或冻结的正常核对事实。压力证据提取的事实仅用于回答生成前的准入。']
    if bad==0 and false==0 and unknown==0:
        lines += ['','本轮四种场景仍然过于容易：未观察到质量下降或误报。它补充了有限的证据顺序、去重和干扰稳健性证据，不能证明真实流量下误报率低，也不能证明累计检测优于单次核对。',
          '下一步应扩大问题结构，例如同表跨行列比较、计算和跨段约束；先锁定问题及可核验的事实或计算规则，再收集全新正常回答。']
    else:
        lines += ['','优先定位出现质量下降、核对分歧或误报的字段、单位与表格轴。保留本轮原结果，修正方案应在后续独立批次验证。']
    lines += ['', '## 方法与限制','',
      '- 原始与反转使用完全相同的原文片段；去重按原文坐标求并集，保留所有原始来源位置。',
      '- 干扰片段取自同篇论文，使用公开题干和字段的字面相似度一次选定，保留原文表头、表题与数值。',
      '- 原始、反转、去重正文至多 6000 字符；干扰条件至多 8000 字符。因此干扰条件同时改变长度与相关背景。',
      '- 所有压力条件在新回答前盲提取并双轮检查，与冻结正常字段及单位一致后才准入；失败不删档。',
      '- 17 道题和论文此前用于开发。所有答案为本轮新生成；同题、多题同论文存在相关性。',
      '- 自动标准事实评价和核对均使用同一模型供应商，双轮一致不等于独立真值；MinerU 源文未经过本轮人工复核。',
      '- 本轮是上下文压力试验，没有运行新的变点流，不给出真实流量误报率或 e-detector 理论保证。',
      f"- 实际返回模型：{s['actual_models']}；API 返回 {s['api_responses']} 次，token {s['total_tokens']}。",'',
      '![正常证据压力结果](normal_stress_quality.png)','']
    (output/'normal_stress_report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    fig, ax = plt.subplots(figsize=(8,4.5))
    x = list(range(len(s['metrics'])))
    ax.bar(x,[r['complete'] for r in s['metrics']],label='完整',color='#298777')
    ax.bar(x,[r['bad'] for r in s['metrics']],bottom=[r['complete'] for r in s['metrics']],label='错误或遗漏',color='#d58a35')
    ax.bar(x,[r['quality_u'] for r in s['metrics']],bottom=[r['complete']+r['bad'] for r in s['metrics']],label='未确定',color='#9e9e9e')
    for i,r in enumerate(s['metrics']):
        ax.text(i,r['n']+.2,f"误报 {r['complete_false_flags']}/{r['complete']}",ha='center',fontsize=10)
    ax.set_xticks(x,[labels[r['condition']] for r in s['metrics']])
    ax.set_ylim(0,max(r['n'] for r in s['metrics'])+4)
    ax.set_ylabel('本轮新回答数')
    ax.set_title('V11：保留原始事实的正常证据压力实验')
    ax.legend(loc='upper right',frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(output/('normal_stress_quality.'+ext),dpi=180)
    plt.close(fig)
    print('V11 report and figure saved',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    report(ap.parse_args().output)
