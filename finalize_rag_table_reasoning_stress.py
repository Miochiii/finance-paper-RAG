"""Offline completion checks and reports; frozen V12/V12.1 methods stay intact.

The filtered-result literal check is explicitly post hoc. It supplements the
pre-generation arithmetic literal check and never changes primary LLM grades.
No function in this module calls an external service.
"""
import argparse
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import rag_table_reasoning_stress as base
from rag_fresh_change_experiment import load, dump, sha, utc, write_csv
from rag_live_evidence_pairs import metrics
from rag_prospective_coverage_validation import verify_hashes


def replay_screening(output, tasks, banks):
    rows = []
    for qid, task in tasks.items():
        bid = task['bank_id']; bank = banks[bid]
        candidate = load(output/'candidate_checks'/(qid+'.json'))['qualified']
        conditions = {c: bank['variants'][c]['structurally_valid'] and
            load(output/'grid_checks'/(bid+'_'+c+'.json'))['plan']['basis_complete']
            for c in base.CONDITIONS}
        rows.append(dict(question_key=qid, source=bank['source'],
            task_type=task['rule']['kind'], candidate_status='pending',
            candidate_two_checks=int(candidate), normal_grid=int(conditions[base.CONDITIONS[0]]),
            distractor_grid=int(conditions[base.CONDITIONS[1]]),
            experimental_admission=int(candidate and all(conditions.values()))))
    if rows != load(output/'candidate_screening.json'):
        raise ValueError('screening admission changed')
    return rows


def expected_cases(screening, questions, tasks, banks, seed):
    pending = [(r['question_key'], c) for r in screening if r['experimental_admission']
               for c in base.CONDITIONS]
    random.Random(seed).shuffle(pending)
    cases = {}; keys = {}
    for i, (qid, condition) in enumerate(pending, 1):
        aid = f'R{i:04d}'
        context = banks[tasks[qid]['bank_id']]['variants'][condition]['context']
        cases[aid] = dict(question=questions[qid]['question'], context=context,
                          context_sha256=sha(context))
        keys[aid] = dict(question_key=qid, condition=condition)
    return cases, keys


def verify_generation_metadata(generation):
    raw = generation['raw_response']; choice = raw['choices'][0]
    for field, actual in [('actual_model', raw['model']), ('response_id', raw['id']),
            ('finish_reason', choice['finish_reason']), ('usage', raw['usage']),
            ('answer', choice['message']['content'].strip())]:
        if generation[field] != actual:
            raise ValueError('generation metadata differs from raw response: '+field)
    if datetime.fromisoformat(generation['started_at']) > datetime.fromisoformat(generation['finished_at']):
        raise ValueError('generation timestamps reversed')


def verify_scope_and_chronology(api, generations, authorization, qualification_time):
    admission = {n: r for n, r in api.items() if n.startswith(('candidate_check_', 'grid_extract_', 'grid_check_'))}
    scoring = {n: r for n, r in api.items() if n not in admission}
    if (len(admission) > authorization['maximum_admission_requests'] or
        len(generations) > authorization['maximum_fresh_answers'] or
        len(api)+len(generations) > authorization['maximum_logical_requests_including_repairs']):
        raise ValueError('authorized scope exceeded')
    for r in admission.values():
        if datetime.fromisoformat(r['started_utc']) >= datetime.fromisoformat(qualification_time):
            raise ValueError('admission API after qualification lock')
    for name, record in scoring.items():
        match = re.search(r'(R\d{4})_r[12]$', name)
        if not match or match[1] not in generations:
            raise ValueError('unrecognized scoring record')
        if datetime.fromisoformat(record['started_utc']) < datetime.fromisoformat(generations[match[1]]['finished_at']):
            raise ValueError('scoring before matching generation finished')
    if not all(r.get('ok') for r in api.values()):
        raise ValueError('failed API records present')
    return dict(admission_requests=len(admission), fresh_answers=len(generations),
                scoring_requests=len(scoring), logical_request_records=len(api)+len(generations))


def completion_audit(output):
    protocol = base.verify_ready(output)
    if load(output/'integrity_audit.json')['status'] != 'passed':
        raise ValueError('primary replay audit required')
    for directory, name in [(output, 'preparation_checkpoint_manifest.json'),
            (Path(protocol['failed_preparation']), 'preparation_checkpoint_manifest.json'),
            (Path(protocol['previous']), 'artifact_manifest.json')]:
        frozen = load(directory/name)
        verify_hashes(frozen['output_hashes']); verify_hashes(frozen['code_and_input_hashes'])
    verify_hashes(protocol['ranking_source_hashes'])
    diagnostic_lock = load(output/'calculation_diagnostic_lock.json')
    verify_hashes(diagnostic_lock['hashes'])
    questions = load(output/'public_questions.json'); tasks = load(output/'private_task_rules.json')
    banks = load(output/'table_banks.json')
    rows = replay_screening(output, tasks, banks)
    admitted = [r for r in rows if r['experimental_admission']]
    expected, keys = expected_cases(rows, questions, tasks, banks, protocol['config']['seed'])
    if expected != load(output/'public_cases.json') or keys != load(output/'private_case_key.json'):
        raise ValueError('randomized cases differ from frozen public rules')
    preflight = load(output/'preflight_lock.json')
    gate = dict(questions=len(admitted) >= protocol['minimum_paired_questions'],
        sources=len({r['source'] for r in admitted}) >= protocol['minimum_paired_sources'],
        task_types=all(sum(r['task_type'] == t for r in admitted) >= protocol['minimum_questions_per_task_type'] for t in base.TYPES))
    if not all(gate.values()) or preflight['checks'] != gate or preflight['admitted'] != [r['question_key'] for r in admitted] or preflight['new_generations'] != len(expected):
        raise ValueError('preflight gate or exclusions changed')
    api = {p.stem: load(p) for p in (output/'api_records').glob('*.json')}
    gens = {p.stem: load(p) for p in (output/'generations').glob('*.json')}
    qualification = load(output/'qualification_lock.json')['locked_at']
    scope = verify_scope_and_chronology(api, gens, load(output/'expanded_api_authorization.json'), qualification)
    if scope['admission_requests'] != 2*len(questions)+3*sum(b['variants'][c]['structurally_valid'] for b in banks.values() for c in base.CONDITIONS):
        raise ValueError('admission request attrition')
    if set(gens) != set(expected):
        raise ValueError('generation attrition')
    for g in gens.values():
        verify_generation_metadata(g)
    first_generation = min(datetime.fromisoformat(g['started_at']) for g in gens.values())
    if datetime.fromisoformat(diagnostic_lock['locked_at']) >= first_generation:
        raise ValueError('numeric diagnostic frozen after answers')
    for directory in ['candidate_checks', 'grid_checks']:
        for path in (output/directory).glob('*.json'):
            if datetime.fromisoformat(load(path)['finished_at']) >= datetime.fromisoformat(qualification):
                raise ValueError('qualification locked before admission completed')
    if not datetime.fromisoformat(qualification) <= datetime.fromisoformat(preflight['locked_at']) < first_generation:
        raise ValueError('preflight chronology invalid')
    models = dict(Counter([r['model_returned'] for r in api.values()]+[g['actual_model'] for g in gens.values()]))
    tokens = sum(r['usage']['total_tokens'] for r in list(api.values())+list(gens.values()))
    summary = load(output/'analysis_summary.json')
    if summary['actual_models'] != models or summary['total_tokens'] != tokens or summary['api_responses'] != scope['logical_request_records']:
        raise ValueError('usage aggregate changed')
    result = dict(status='passed', checked_at=utc(), **scope,
        candidate_screening_and_randomization_replayed=True, source_ranking_corpus_hashes_checked=len(protocol['ranking_source_hashes']),
        all_generation_metadata_equal_raw_responses=True, all_scoring_after_matching_answer=True,
        numeric_diagnostic_frozen_before_answers=True, earlier_frozen_checkpoints_unchanged=True,
        qualified_questions=len(admitted), qualified_target_sources=len({r['source'] for r in admitted}),
        excluded_before_answers=len(rows)-len(admitted), authorized_logical_limit=810,
        actual_network_attempts_not_inferred_from_logical_records=True,
        scope='protocol_integrity_and_authorization_not_independent_semantic_truth')
    dump(output/'completion_integrity_audit.json', result)
    return result


def numbered_fields(answer):
    """Post hoc conservative reading, without gold or entity vocabulary."""
    found = {1: [], 2: []}
    for line in unicodedata.normalize('NFKC', answer).splitlines():
        line = line.strip().replace('**', '').replace('`', '')
        match = re.match(r'^(?:([12])\s*[.)、]|\(([12])\))\s*(.+)$', line)
        if match:
            index = int(match[1] or match[2]); value = match[3]
            if ':' in value: value = value.rsplit(':', 1)[1].strip()
            found[index].append(value)
    if any(len(v) != 1 for v in found.values()):
        return None
    entity, literal = found[1][0], found[2][0]
    entity = re.sub(r'^组合(?=\d+$)', '', entity)
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*%?', literal)
    return (entity, match[1]) if match else None


def rational_filter_winner(rule, grid):
    """Separate Fraction filter/max implementation, without calculate()."""
    comparisons = {'>': lambda a,b: a>b, '>=': lambda a,b: a>=b,
                   '<': lambda a,b: a<b, '<=': lambda a,b: a<=b}
    eligible = []
    for row in grid:
        values = {k: v['value'] for k,v in row['values'].items()}
        passed = all(values[name] == threshold if op == '==' else
                     comparisons[op](Fraction(values[name]), Fraction(threshold))
                     for name,op,threshold in rule['filters'])
        if passed: eligible.append((row['entity'], Fraction(values[rule['target']])))
    if not eligible: raise ValueError('no eligible entities')
    maximum = max(v for _,v in eligible)
    winners = [(e,v) for e,v in eligible if v == maximum]
    if len(winners) != 1: raise ValueError('maximum not unique')
    return winners[0]


def filtered_diagnostic(output):
    keys = load(output/'private_case_key.json'); tasks = load(output/'private_task_rules.json')
    banks = load(output/'table_banks.json'); refs = load(output/'private_references.json'); rows = []
    for aid,key in keys.items():
        qid = key['question_key']; task = tasks[qid]
        if task['rule']['kind'] != 'filtered_max': continue
        winner, value = rational_filter_winner(task['rule'], banks[task['bank_id']]['private_source_grid'])
        if winner != refs[qid]['facts'][0]['expected'] or value != Fraction(refs[qid]['facts'][1]['expected']):
            raise ValueError('separate rational filter disagrees with frozen reference')
        observed = numbered_fields(load(output/'generations'/(aid+'.json'))['answer'])
        matched = observed is not None and observed[0] == winner and Fraction(observed[1]) == value
        rows.append(dict(aid=aid, **key, read_status='readable' if observed else 'U',
            entity_and_value_match=int(matched) if observed else 'U',
            primary_quality=load(output/'scores'/(aid+'.json'))['quality']))
    summary = dict(status='completed', checked_at=utc(), post_hoc=True, primary_grades_changed=False,
        calculated_answers=len(rows), readable=sum(r['read_status']=='readable' for r in rows),
        unreadable_U=sum(r['read_status']=='U' for r in rows),
        literal_matches=sum(r['entity_and_value_match']==1 for r in rows),
        literal_mismatches=sum(r['entity_and_value_match']==0 for r in rows),
        primary_complete_but_literal_mismatch=sum(r['primary_quality']=='2' and r['entity_and_value_match']==0 for r in rows),
        separate_Fraction_filter_max_verified_all_references=True,
        limitation='post_hoc_simple_numbered_fields_only_not_a_frozen_primary_evaluator')
    write_csv(output/'filtered_literal_diagnostic.csv', rows)
    dump(output/'filtered_literal_diagnostic.json', summary)
    return summary


def report(output):
    if load(output/'completion_integrity_audit.json')['status'] != 'passed': raise ValueError('completion audit required')
    s = load(output/'analysis_summary.json'); literal = load(output/'calculation_literal_diagnostic.json')
    filtered = load(output/'filtered_literal_diagnostic.json'); tasks = load(output/'private_task_rules.json')
    banks = load(output/'table_banks.json'); keys = load(output/'private_case_key.json')
    answer_rows = []
    for aid,key in keys.items():
        task = tasks[key['question_key']]; score = load(output/'scores'/(aid+'.json'))
        trace = banks[task['bank_id']]['distractor_selection_trace']
        answer_rows.append(dict(aid=aid, **key, task_type=task['rule']['kind'],
            direct_fields=bool(trace['direct_field_hits']), quality=score['quality'],
            proxy=score['proxy_comparison']['signal']['status'], refusal=score['field_refusal']['refusal']))
    strata = [dict(distractor_match=label, condition=c, **metrics([r for r in answer_rows if r['condition']==c and r['direct_fields']==strong]))
              for label,strong in [('direct_field',True), ('alias_only',False)] for c in base.CONDITIONS]
    write_csv(output/'distractor_stratified_metrics.csv', strata)
    admitted_banks = {tasks[k['question_key']]['bank_id'] for k in keys.values()}
    same = sum(banks[b]['distractor_selection_trace']['same_paper'] for b in admitted_banks)
    direct = sum(bool(banks[b]['distractor_selection_trace']['direct_field_hits']) for b in admitted_banks)
    lines = ['# V12.1 表格计算与多条件查找：结果','',
        f"27 道候选全部通过双轮题目/来源/计算检查。正常及干扰网格 16/18 组通过；回答生成前排除 6 道，最终 21 道、7 篇目标论文、42 条全新配对回答。候选保持 pending。",'',
        '| 条件 | 回答 | 自动评分完整 | 错误/遗漏 | 错误检出 | 完整回答误报 | 监测 U |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['metrics']:
        lines.append(f"| {r['condition']} | {r['n']} | {r['complete']} | {r['bad']} | {r['bad_flagged']} | {r['complete_false_flags']} | {r['proxy_u']} |")
    lines += ['', '## 错误类型与自动核验','',
        f"- 比值计算出现 {literal['numeric_mismatches']} 条数值错误；28 条差值/比值结果均可读取，其中 25 条与四位小数规则一致。该字面检查在回答生成前冻结。差值 14/14 完整。",
        f"- 多条件筛选出现 {filtered['literal_mismatches']} 条实体及对应值错误。另用 Fraction 独立实现筛选/最大值并核对 14 条编号字段；这是事后补充诊断，没有改变主评分。",
        '- 7 条错误全部未拒答，监测侧全部标出。四舍五入误差按题目明确的四位小数要求计错，不能把这 3 条与严重事实错误混为一谈。',
        '- 1 条正确筛选回答的监测 U：两轮对未重复标注百分号分别判 covered 和 partial。保留原始 U，不重评分、不删除。',
        '- 正常条件自身有 4/21 条错误；干扰条件有 3/21 条。21 个配对中 1 个变差、2 个变好、18 个不变；单次相关配对不能证明干扰改善质量。', '',
        '## 干扰和准入限制','',
        f'- 准备阶段 7 张同篇、2 张跨篇完整表；准入后 {same} 张同篇、{len(admitted_banks)-same} 张跨篇。优先同篇、按公开字段和别名排序，没有改变原文。',
        f'- 准入后 {direct} 张直接字段匹配表、{len(admitted_banks)-direct} 张仅宽泛别名匹配表。别名组含算法原理、决策树参数、指标分类表，不能全部称为强实验干扰；分层结果另存 CSV。',
        '- T01 干扰网格的多字段数值正确，但转置表引文只覆盖错误率行，AUC/准确率缺乏对应引文；T04 干扰网格的一轮单位检查返回 U。各关联三题在生成前排除，不能算成回答错误，也不能据此说六道题真值无效。',
        '- 6/27 候选被网格准入排除，结果只覆盖其余 7 篇目标论文。上下文短且原始目标表完整，不代表长上下文或未见论文。','',
        '## 对变点实验的含义','',
        '本轮获得了证据完整时自发产生、且没有拒答的质量错误，提供了更有用的基线样本。参考评价与监测侧仍使用同供应商，计算规则部分共享；7/7 检出是本批观察值，不能推出真实检出率 100%。',
        '0/35 次完整回答误报中有 1 条未确定；确定的完整回答是 34 条。这里的真实正常质量错误是有效检出，不是监测误报；序列级变点报警仍需在稳定的非零基线错误率下校准。',
        '本轮没有运行新变点序列，也没有调整检测器或阈值。不能据此宣称累计变点方法优于首个事实核对阳性。','',
        '## 后续顺序','',
        '1. 先在新版本明确单位规则：题干已限定百分数展示值时如何处理回答缺省单位；转置表引文按涉及的全部行保存。保留 V12.1 评分与排除记录，用新题和新回答验证修改。',
        '2. 建立按论文分组的独立基线批次，覆盖计算、约束筛选和直接事实问答，并区分强字段匹配干扰与一般表格背景；当前批次仅用于开发诊断。',
        '3. 在新基线冻结后校准序列阈值，再收集独立稀疏退化序列。比较 e-detector、CUSUM、首个核对阳性和首个拒答的序列误报与延迟；题库结构稳定时才将质量变化解释为系统退化。',
        '4. 7 条错误涉及 5 道独立题、5 篇目标论文，同题配对不当作七次独立成功。下一批优先扩大论文和问题覆盖，不重复本批刷检出率。','',
        f"实际模型均为 deepseek-flash。准入 108 次、生成 42 次、评分 168 次，总计 {s['api_responses']} 条逻辑请求返回，token {s['total_tokens']}；失败记录 0，低于本轮授权上限 810。物理网络重试次数不能从返回记录推断。",
        '主派生重放及补充完整性审计通过。既有冻结实验保持不变；原始题目、证据、标准事实和 API 返回留在工作项目。','']
    (output/'table_reasoning_report.md').write_text('\n'.join(lines), encoding='utf-8')
    conclusion = ['# V12.1 本轮结论','',
        '21 道新题、7 篇目标论文、42 条全新回答。自动评分完整 35 条、错误 7 条；7 条错误均被检出，0 条完整回答误报，1 条监测未确定。',
        '错误包括 3 条比值/四舍五入错误和 4 条筛选实体错误；离线字面与确定性运算补充核验支持这些分类。', '',
        '这轮的进展是得到更难任务中的自发质量错误，以及正常证据完整时的非零错误基线。检测算法没有改变，检出样本仍少，尚不能证明变点效果改善。', '',
        '下一步先明确缺省单位和转置表引文规则，再以新的论文分组批次收集独立基线、校准序列阈值。正常单点错误不应直接等同于变点。', '',
        '详细结果与限制见 table_reasoning_report.md。']
    (output/'table_reasoning_conclusions.md').write_text('\n'.join(conclusion)+'\n', encoding='utf-8')
    workflow = dict(status='completed_v12_1', completed_at=utc(), questions=21, target_sources=7,
        fresh_answers=42, automatic_complete=35, quality_errors=7, bad_flagged=7,
        complete_false_flags=0, quality_U=0, monitor_U=1,
        arithmetic_literal_check='frozen_before_answers', filtered_literal_check='post_hoc_supplement',
        total_logical_API_records=s['api_responses'], actual_models=s['actual_models'],
        candidates_still_pending=True, earlier_results_unchanged=True, no_detector_or_threshold_change=True,
        new_change_point_sequence_executed=False, next_stage='new_version_unit_and_quote_rules_then_independent_grouped_baseline',
        git_commit=False, github_push=False)
    dump(output/'workflow_summary.json', workflow)
    return workflow


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = load(output/'analysis_summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), layout='constrained')
    labels = ['Original table', 'Extra intact table']
    for i,r in enumerate(s['metrics']):
        axes[0].bar(i, r['complete'], color='#3c827d', label='Complete' if i==0 else None)
        axes[0].bar(i, r['bad'], bottom=r['complete'], color='#cb6855', label='Wrong' if i==0 else None)
        axes[0].text(i, r['complete']/2, str(r['complete']), ha='center', va='center', color='white')
        axes[0].text(i, r['complete']+r['bad']/2, str(r['bad']), ha='center', va='center', color='white')
    axes[0].set(xticks=[0,1], xticklabels=labels, ylabel='Fresh answers', ylim=(0,26), title='Primary automatic quality (21 pairs)')
    axes[0].legend(loc='upper center', ncol=2, frameon=False)
    for i,c in enumerate(base.CONDITIONS):
        counts = [next(r['bad'] for r in s['task_metrics'] if r['condition']==c and r['task_type']==t) for t in base.TYPES]
        axes[1].bar([x+(i-.5)*.32 for x in range(3)], counts, width=.32, color=['#4b79aa','#bf9451'][i], label=labels[i])
        for x,count in enumerate(counts):
            axes[1].text(x+(i-.5)*.32, count+.08, str(count), ha='center', fontsize=9)
    axes[1].set(xticks=[0,1,2], xticklabels=['Difference','Ratio','Filtered max'], ylabel='Wrong answers', ylim=(0,4.5), title='Errors by task (7 pairs each)')
    axes[1].legend(loc='upper center', frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('V12.1: 7/7 observed errors flagged; 0 false flags; 1 monitor U', fontsize=12)
    fig.savefig(output/'table_reasoning_summary.png', dpi=180)
    fig.savefig(output/'table_reasoning_summary.svg')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    output = ap.parse_args().output
    completion_audit(output)
    filtered_diagnostic(output)
    workflow = report(output)
    plot(output)
    print('V12.1 completed:', workflow, flush=True)
