"""Aggregate V12 table reasoning report; source/formula limits are explicit."""
import argparse
from pathlib import Path
from rag_fresh_change_experiment import load


def report(output):
    if load(output/'integrity_audit.json')['status']!='passed':raise ValueError('audit required')
    s=load(output/'analysis_summary.json')
    lines=['# V12 新题：表格计算与多条件查找','',
      f"候选 {s['candidate_questions']} 道；准入 {s['qualified_questions']} 道、{s['qualified_sources']} 篇来源；回答生成前排除 {s['excluded_before_answers']} 道，全部留档。",
      f"新回答 {s['fresh_generations']} 条。题库仍为 pending，本轮准入不等于正式评测集晋升。",'',
      '| 条件 | 新回答 | 完整 | 错误/遗漏 | 质量 U | 完整回答误报 | 核对 U |','|---|---:|---:|---:|---:|---:|---:|']
    for r in s['metrics']:
        lines.append(f"| {r['condition']} | {r['n']} | {r['complete']} | {r['bad']} | {r['quality_u']} | {r['complete_false_flags']} | {r['proxy_u']} |")
    lines+=['','## 方法','',
      '- 新题结构包括差值、比值、多条件筛选最大值；操作数有原文行列与精确字符坐标。',
      '- 四位小数使用 Decimal 的 ROUND_HALF_UP；用 Fraction 独立复核四舍五入结果。百分数展示值相减按百分点输出。',
      '- 参考侧从原始表格单元格确定性计算；核对侧独立盲提取正常表格网格后计算。双方共享公开计算规则。',
      '- 候选先双轮检查来源与计算；原始和干扰证据均盲提取并双轮检查，失败在回答前留档。',
      '- 干扰只选同篇论文的另一张完整实验表，按公开目标字段和对象排序；不使用论文标题或标准值。',
      '- 真实正常回答质量错误与核对误报分别统计；所有 U 保留。没有调整变点检测器和阈值。','',
      '## 结论范围','',
      '这些论文此前用于开发，问题规则在读过来源后设计。新问题和新回答不构成未见论文上的泛化验证。',
      '参考和核对使用相同算术实现与同一供应商的语义评价，双轮一致不构成独立真值。',
      '本轮只分析点级质量与误报；尚未运行新变点序列。',
      f"实际模型：{s['actual_models']}；API 返回 {s['api_responses']} 次；token {s['total_tokens']}。",'']
    (output/'table_reasoning_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('V12 report saved',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    report(ap.parse_args().output)
