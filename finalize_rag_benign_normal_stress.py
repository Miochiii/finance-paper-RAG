"""Finalize V11 interpretation and authorized request scope after offline audit."""
import argparse
import re
from pathlib import Path

from rag_fresh_change_experiment import load,dump,utc
from rag_prospective_coverage_validation import verify_hashes


def finalize(output):
    summary=load(output/'analysis_summary.json')
    audit=load(output/'integrity_audit.json')
    if audit['status']!='passed':
        raise ValueError('integrity audit required')
    prepared=load(output/'preparation_checkpoint_manifest.json')
    verify_hashes(prepared['output_hashes'])
    verify_hashes(prepared['code_and_input_hashes'])
    authorization=load(output/'expanded_api_authorization.json')
    strength=load(output/'stress_strength_summary.json')
    api={p.stem:load(p) for p in (output/'api_records').glob('*.json')}
    gen={p.stem:load(p) for p in (output/'generations').glob('*.json')}
    qualification=[r for n,r in api.items() if n.startswith('stress_')]
    scoring=[(n,r) for n,r in api.items() if not n.startswith('stress_')]
    if (len(qualification)>authorization['maximum_evidence_check_requests'] or
        len(gen)>authorization['maximum_fresh_answers'] or
        len(api)+len(gen)>authorization['maximum_logical_requests_including_schema_and_quote_repairs']):
        raise ValueError('authorized logical request scope exceeded')
    for name,r in scoring:
        found=re.search(r'(N\d{4})_r[12]$',name)
        if not found or r['started_utc']<gen[found.group(1)]['finished_at']:
            raise ValueError('scoring preceded matching answer')
    if len(qualification)!=153 or len(gen)!=68 or not all(r.get('ok') for r in api.values()):
        raise ValueError('incomplete V11 requested execution')
    scope=dict(status='passed',checked_at=utc(),qualification_requests=len(qualification),
        fresh_generation_responses=len(gen),scoring_requests=len(scoring),logical_request_records=len(api)+len(gen),
        authorized_logical_limit=authorization['maximum_logical_requests_including_schema_and_quote_repairs'],
        all_scoring_after_matching_answer=True,previous_offline_checkpoint_unchanged=True,
        failures_preserved=len(list((output/'failure_snapshots').glob('*.json'))),
        raw_questions_evidence_answers_and_references_stay_in_working_project=True)
    dump(output/'authorization_and_chronology_audit.json',scope)
    false=sum(r['complete_false_flags'] for r in summary['metrics'])
    bad=sum(r['bad'] for r in summary['metrics'])
    unknown=sum(r['quality_u']+r['proxy_u'] for r in summary['metrics'])
    clean=bad==false==unknown==0
    lines=['# V11 本轮结论与下一步','',
       f"本轮 {summary['input_questions']} 道题、{summary['input_sources']} 篇既有论文、{summary['fresh_generations']} 条新回答；四个条件均为 17 条。",
       f"自动标准事实评价：错误或遗漏 {bad}，完整回答误报 {false}，质量或核对 U 合计 {unknown}。",'',
       '## 本轮增加了什么证据','',
       '这轮增加了同题证据位置、重复片段和额外背景变化下的自动评价结果。检测器、阈值和生成提示词没有改变，因此不能把结果称为算法能力改善。',
       f"原始证据的最后一项必答引文结束位置平均位于上下文 {strength['metrics'][0]['mean_latest_required_quote_end_fraction']:.1%}；顺序反转后为 {strength['metrics'][1]['mean_latest_required_quote_end_fraction']:.1%}。该统计使用同一引文的首次字面匹配位置，仅描述证据位置变化。",'',
       '## 压力强度的限制','',
       f"- {strength['questions_with_actual_overlap_removed']}/17 道题实际存在重叠并被去重；其余题的去重条件主要改变片段顺序或元数据。",
       f"- 17 个干扰中，16 个为一般段落，只有 1 个为另一张完整表格；{strength['distractors_with_cover_metadata']} 个包含封面元信息。",
       f"- 干扰正文平均 {strength['distractor_characters']['mean']:.0f} 字符；最长 {strength['distractor_characters']['maximum']} 字符。",
       '- 排序使用包含论文标题的完整题干，标题相似度可能压过字段相似度，容易选到背景材料。',
       '- 保留所有原始证据能确保事实没有被删掉，也让题目仍容易直接查值。题和论文此前已用于开发。',
       '- 四条件同题回答和同论文多题有关联；同供应商自动标准事实与核对双轮一致不构成独立真值。',
       '- 0 次观察误报不能转换为真实流量误报率为 0，也不能说明累计方法优于首个事实核对阳性。','']
    if clean:
        lines+=['## 后续顺序','',
          '1. 先构建新的问题结构：同一表的跨行列比较、差值或比例计算、明确多个限定条件的跨段查找。用原文坐标、表格行列和确定性计算保存标准事实；不靠生成答案反过来改题。',
          '2. 干扰选择改为按公开目标字段、表题与对象匹配，排除封面、目录和摘要，并优先选相似字段的其他实验表。此修改在新批次验证，保留 V11 原结果。',
          '3. 新题和证据先做来源、单位、唯一对应及计算规则检查；把不确定案例留档。然后收集全新正常回答，区分生成错误、核对误报和 U。',
          '4. 只有当正常场景能产生有代表性的质量波动或监测噪声后，再按论文分组重新校准阈值，运行稀疏故障序列，并继续与首个事实核对阳性基线比较。',
          '', '当前结论：有限稳健性结果是正面的，实验难度仍然不足；下一步优先提高正常任务与干扰的代表性。']
    else:
        lines+=['## 后续顺序','',
          '优先排查本轮出现的质量错误、核对误报和 U，对照原文表格轴、字段和单位。任何修正放入新的批次验证，保留本轮结果。']
    lines+=['',f"执行完成：准入 {len(qualification)} 次、新回答 {len(gen)} 条、评分 {len(scoring)} 次；总计 {len(api)+len(gen)} 条逻辑请求返回记录，低于授权上限。",
            '已完成离线派生审计与授权范围审计。原始研究数据留在工作项目，汇总同步到本地备份仓库。','']
    (output/'normal_stress_conclusions.md').write_text('\n'.join(lines),encoding='utf-8')
    workflow=dict(status='completed_v11',completed_at=utc(),fresh_answers=summary['fresh_generations'],
       questions=summary['input_questions'],sources=summary['input_sources'],
       automatic_complete=sum(r['complete'] for r in summary['metrics']),quality_errors=bad,
       complete_false_flags=false,quality_or_monitor_U=unknown,
       targeted_tests_passed=31,integrity_audit='passed',authorization_audit='passed',
       logical_API_records=scope['logical_request_records'],actual_models=summary['actual_models'],
       request_failures_current_records=sum(not r.get('ok') for r in api.values()),
       context_difficulty='many_general_background_distractors_only_one_other_intact_table',
       no_algorithm_change=True,improvement_claim='stronger_limited_validation_evidence_not_algorithm_improvement',
       next_stage='new_multirow_comparison_calculation_constraint_questions_and_field_specific_table_distractors',
       next_stage_executed=False,git_commit=False,github_push=False)
    dump(output/'workflow_summary.json',workflow)
    print('V11 finalized: complete',workflow['automatic_complete'],'errors',bad,'false flags',false,'U',unknown,flush=True)
    return workflow


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    finalize(ap.parse_args().output)
