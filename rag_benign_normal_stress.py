"""V11: source-preserving context stress, with qualification before fresh answers.

No detector or prompt tuning. Construction sees only public question fields,
source text and cached retrieval coordinates; references are used for admission
and evaluation only. All qualification failures are retained.
"""
import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
import rag_field_retrieval_trial as experiment
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load, dump, digest, sha, utc, write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_subtle_fault_pilot import batch
from rag_mineru_coverage_v4 import source_context, evidence_projection
from rag_objective_slots import EXTRACT_SLOTS, CHECK_SLOTS, exact_source_quote
from rag_percent_fact_guard import percent_unit_proxy, fact_projection_issues
from rag_live_evidence_pairs import same_slot, metrics, CHECK_SUFFIX, COMMON_MODULES

CONDITIONS = ['live_normal', 'live_reverse', 'live_dedup', 'live_distractor']
NEW_CODE = ['rag_benign_normal_stress.py', 'audit_rag_benign_normal_stress.py',
            'report_rag_benign_normal_stress.py', 'tests/test_rag_benign_normal_stress.py']
CONFIG = dict(seed=2026100211, distractor_max_characters=1800,
              maximum_context_text_characters=8000, workers=4)
TABLE = re.compile(r'\[TABLE_START\].*?\[/TABLE_END\]', re.S)


def union_intervals(rows):
    spans = sorted((r['start'], r['end']) for r in rows)
    out = []
    for a, b in spans:
        if a < 0 or b <= a:
            raise ValueError('invalid source coordinates')
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def source_row(text, source, a, b, kind):
    if not 0 <= a < b <= len(text):
        raise ValueError('source span outside text')
    return dict(source=source, start=a, end=b, text=text[a:b],
                text_sha256=sha(text[a:b]), kind=kind)


def public_grams(question):
    if set(question) != {'question', 'requested_slots', 'sources'}:
        raise ValueError('construction requires public question only')
    text = auto.norm(question['question'] + ' ' + ' '.join(s['label'] for s in question['requested_slots']))
    return {text[i:i+2] for i in range(len(text)-1) if not text[i:i+2].isspace()}


def select_distractor(question, text, source, occupied, maximum):
    """Rank intact tables/paragraphs outside original passages using public text.

    The most similar eligible passage is selected once, before any test answer.
    Complete tables keep captions and headers. Paragraphs cannot cut a table.
    """
    grams = public_grams(question)
    tables = list(TABLE.finditer(text))
    candidates = [(m.start(), m.end(), 'intact_table') for m in tables]
    candidates += [(m.start(), m.end(), 'paragraph') for m in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]+)*', text)
                   if not any(m.start() < t.end() and t.start() < m.end() for t in tables)]
    scored = []
    for a, b, kind in candidates:
        if not 120 <= b-a <= maximum or any(a < d+250 and c-250 < b for c, d in occupied):
            continue
        t = auto.norm(text[a:b])
        overlap = len(grams & {t[i:i+2] for i in range(len(t)-1)})
        if overlap:
            scored.append((overlap, kind == 'intact_table', -(b-a), -a, a, b, kind))
    if not scored:
        return None, dict(candidate_count=0, rule='public_bigrams_outside_baseline_plus_250')
    best = max(scored)
    _, _, _, _, a, b, kind = best
    return source_row(text, source, a, b, 'distractor_'+kind), dict(
        candidate_count=len(scored), public_bigram_hits=best[0], start=a, end=b,
        rule='public_bigrams_then_intact_table_then_shorter_then_earlier')


def construct(question, retrieval, text, cfg=CONFIG):
    public_grams(question)  # Reject private keys even for pure permutations.
    source = question['sources'][0]
    rows = retrieval['retrieved']
    if len(question['sources']) != 1 or any(r['source'] != source or
         r['text'] != text[r['start']:r['end']] for r in rows):
        raise ValueError('cached passages do not match exact MinerU source')
    occupied = union_intervals(rows)
    dedup = [source_row(text, source, a, b, 'merged_source_interval') for a, b in occupied]
    distractor, trace = select_distractor(question, text, source, occupied, cfg['distractor_max_characters'])
    choices = {'live_normal': copy.deepcopy(rows), 'live_reverse': copy.deepcopy(rows[::-1]),
               'live_dedup': dedup, 'live_distractor': ([distractor]+dedup if distractor else [])}
    out = {}
    for condition, passages in choices.items():
        size = sum(len(r['text']) for r in passages)
        valid = bool(passages) and size <= cfg['maximum_context_text_characters']
        out[condition] = dict(retrieved=passages, evidence=evidence_projection(passages),
            context=source_context(passages), structurally_valid=valid, text_characters=size,
            reason='passed' if valid else 'no_eligible_distractor' if not passages else 'context_budget',
            baseline_union=occupied, public_selection_trace=trace if condition == 'live_distractor' else {},
            original_text_characters=sum(len(r['text']) for r in rows),
            all_baseline_source_positions_preserved=valid and union_intervals(passages) == occupied
                if condition != 'live_distractor' else bool(distractor) and valid)
    return out


def literal_basis_preserved(plan, evidence):
    return bool(plan['basis_complete']) and all(any(exact_source_quote(s['quote'], e['text'])
               for e in evidence) for s in plan['slots'])


def qualify(normal, extracted):
    slots = extracted['slots']
    return bool(extracted['basis_complete'] and len(slots) == len(normal['slots']) and
                all(same_slot(a, b) for a, b in zip(normal['slots'], slots)))


def prepare(parent, output):
    if (output/'protocol_lock.json').exists():
        verify(output)
        return
    old = load(parent/'artifact_manifest.json')
    verify_hashes(old['output_hashes'])
    verify_hashes(old['code_hashes'])
    prior = load(parent/'protocol_lock.json')
    questions = load(parent/'table_axis_clarification_20261002/next_pair_public_questions.json')
    refs = load(parent/'private_references.json')
    retrieval = {q: dict(load(parent/'public_retrieval.json')[q], **v) for q, v in questions.items()}
    plans = {q: load(parent/('table_axis_clarification_20261002/proxy_plans/clarified.json'
                if q == 'P0002' else 'proxy_plans/'+q+'.json')) for q in questions}
    if fact_projection_issues(refs, plans) or not all(p['basis_complete'] for p in plans.values()):
        raise ValueError('initial normal basis preflight failed')
    sources = {s: load(parent/'source_manifest.json')[s] for v in questions.values() for s in v['sources']}
    texts = {s: blocks_to_text(load(Path(v['mineru_path']))) for s, v in sources.items()}
    for s, v in sources.items():
        if digest(Path(v['mineru_path'])) != v['mineru_sha256'] or sha(texts[s]) != v['text_sha256']:
            raise ValueError('MinerU source hash changed')
    variants = {q: construct(v, retrieval[q], texts[v['sources'][0]]) for q, v in questions.items()}
    for q, bank in variants.items():
        for row in bank.values():
            row['normal_literal_basis_preserved'] = literal_basis_preserved(plans[q], row['evidence'])
            row['structurally_valid'] &= row['normal_literal_basis_preserved']
            if not row['normal_literal_basis_preserved']:
                row['reason'] = 'normal_literal_basis_missing'
    output.mkdir(parents=True, exist_ok=True)
    for n, v in [('public_questions.json', questions), ('private_references.json', refs),
                 ('public_retrieval.json', retrieval), ('source_manifest.json', sources),
                 ('context_variants.json', variants)]:
        dump(output/n, v)
    for q, plan in plans.items():
        dump(output/'proxy_plans'/(q+'.json'), plan)
    inputs = [parent/n for n in ['artifact_manifest.json','protocol_lock.json','public_retrieval.json',
               'source_manifest.json','private_references.json','table_axis_clarification_20261002/next_pair_public_questions.json']]
    inputs += [Path(__file__).parent/n for n in NEW_CODE+COMMON_MODULES+['rag_live_evidence_pairs.py','rag_core/mineru_loader.py']]
    files = [output/n for n in ['public_questions.json','private_references.json','public_retrieval.json',
             'source_manifest.json','context_variants.json']] + sorted((output/'proxy_plans').glob('*.json'))
    dump(output/'protocol_lock.json', dict(version='benign_normal_stress_v11', locked_at=utc(), previous=str(parent.resolve()),
         input_hashes={str(p.resolve()):digest(p) for p in inputs},
         source_hashes={str(Path(v['mineru_path']).resolve()):v['mineru_sha256'] for v in sources.values()},
         initial_hashes={str(p.resolve()):digest(p) for p in files}, config=dict(prior['config'],**CONFIG),
         conditions=CONDITIONS, maximum_new_generations=len(questions)*len(CONDITIONS),
         minimum_admitted_questions_per_condition=10, minimum_admitted_sources_per_condition=6,
         construction='public_only_permutation_union_and_same_paper_distractor_no_character_edits',
         qualification='literal_source_basis_preserved_then_blind_extract_and_two_checks_agree_with_normal_slots',
         monitor='frozen_normal_plans_stress_plans_used_only_for_preanswer_admission',
         attrition='all_preanswer_failures_retained_no_answer_based_selection_or_retry',
         context_budget='baseline_reverse_dedup_at_most_6000_raw_characters_distractor_at_most_8000',
         interpretation='seen_17_question_11_paper_context_stress_pilot_same_vendor_automatic_truth_not_natural_traffic',
         analysis='paired_quality_change_and_false_flags_among_reference_complete_answers_no_detector_tuning'))
    print('V11 prepared',len(questions),'questions;',sum(r['structurally_valid'] for b in variants.values() for r in b.values()),'structural cases',flush=True)


def verify(output):
    m = load(output/'protocol_lock.json')
    for k in ['input_hashes','source_hashes','initial_hashes']:
        verify_hashes(m[k])
    return m


def extract_plan(output, name, question, evidence):
    payload = dict(question=question['question'], requested_slots=question['requested_slots'], retrieved_evidence=evidence)
    raw = auto.request(output, 'stress_extract_'+name, EXTRACT_SLOTS, payload)
    checks = [auto.request(output, f'stress_check_{name}_r{n}', CHECK_SLOTS+CHECK_SUFFIX[n],
              dict(**payload, temporary_slots=raw.get('slots',[])), n) for n in (1,2)]
    return dict(question=question['question'], evidence=evidence, raw=raw, checks=checks,
                **percent_unit_proxy(raw, checks, question, evidence))


def check_contexts(output):
    m = verify(output)
    if (output/'qualification_lock.json').exists():
        verify_hashes(load(output/'qualification_lock.json')['hashes'])
        return
    if list((output/'generations').glob('*.json')):
        raise ValueError('qualification cannot follow answers')
    qs = load(output/'public_questions.json')
    variants = load(output/'context_variants.json')
    def one(item):
        q, c = item
        path = output/'stress_plans'/(q+'_'+c+'.json')
        if path.exists():
            return load(path)['qualified']
        plan = extract_plan(output, q+'_'+c, qs[q], variants[q][c]['evidence'])
        result = qualify(load(output/'proxy_plans'/(q+'.json')), plan)
        dump(path, dict(plan=plan, qualified=result, finished_at=utc()))
        return result
    batch([(q,c) for q, b in variants.items() for c,r in b.items()
           if c != 'live_normal' and r['structurally_valid']], one, m['config']['workers'], 'V11_CONTEXT_CHECK')
    screen = []
    for q, bank in variants.items():
        for c, row in bank.items():
            admitted = row['structurally_valid'] and (c == 'live_normal' or
                       load(output/'stress_plans'/(q+'_'+c+'.json'))['qualified'])
            screen.append(dict(question_key=q, condition=c, source=qs[q]['sources'][0],
              structural=int(row['structurally_valid']), qualified=int(admitted),
              structural_reason=row['reason'], text_characters=row['text_characters'],
              baseline_text_characters=row['original_text_characters']))
    dump(output/'context_screening.json', screen)
    write_csv(output/'context_screening.csv', screen)
    files = [output/'context_screening.json']+sorted((output/'stress_plans').glob('*.json'))+sorted((output/'api_records').glob('*.json'))
    dump(output/'qualification_lock.json', dict(locked_at=utc(), before_answers=True,
            hashes={str(p.resolve()):digest(p) for p in files}))


def freeze_cases(output):
    m = verify(output)
    verify_hashes(load(output/'qualification_lock.json')['hashes'])
    screen = load(output/'context_screening.json')
    checks = {c: dict(questions=sum(r['qualified'] for r in screen if r['condition']==c),
              sources=len({r['source'] for r in screen if r['condition']==c and r['qualified']})) for c in CONDITIONS}
    passed = all(v['questions'] >= m['minimum_admitted_questions_per_condition'] and
                 v['sources'] >= m['minimum_admitted_sources_per_condition'] for v in checks.values())
    if not passed:
        dump(output/'preflight_failure.json', dict(checked_at=utc(),status='failed',checks=checks,generation_prohibited=True))
        raise ValueError('preflight failed; no generation permitted')
    if (output/'preflight_lock.json').exists():
        verify_ready(output)
        return
    if list((output/'generations').glob('*.json')):
        raise ValueError('preflight cannot follow answers')
    qs, variants = load(output/'public_questions.json'), load(output/'context_variants.json')
    pending = [(r['question_key'],r['condition']) for r in screen if r['qualified']]
    random.Random(m['config']['seed']).shuffle(pending)
    cases, keys = {}, {}
    for i,(q,c) in enumerate(pending,1):
        aid = f'N{i:04d}'
        context = variants[q][c]['context']
        cases[aid] = dict(question=qs[q]['question'],context=context,context_sha256=sha(context))
        keys[aid] = dict(question_key=q,condition=c)
    for n,v in [('public_cases.json',cases),('private_case_key.json',keys),('private_control_answers.json',{})]:
        dump(output/n,v)
    for n,files in [('retrieval_lock.json',[output/'public_retrieval.json']),
       ('proxy_lock.json',sorted((output/'proxy_plans').glob('*.json'))),
       ('cases_lock.json',[output/k for k in ['public_cases.json','private_case_key.json','private_control_answers.json','qualification_lock.json']])]:
        dump(output/n,dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in files}))
    files = [output/n for n in ['qualification_lock.json','cases_lock.json','proxy_lock.json','retrieval_lock.json']]
    dump(output/'preflight_lock.json',dict(locked_at=utc(),status='passed',checks=checks,new_generations=len(cases),
         hashes={str(p.resolve()):digest(p) for p in files}))
    print('V11 preflight passed;',len(cases),'fresh answers frozen',flush=True)


def verify_ready(output):
    m = verify(output)
    lock = load(output/'preflight_lock.json')
    if lock['status'] != 'passed':
        raise ValueError('generation prohibited without passed preflight')
    verify_hashes(lock['hashes'])
    experiment.verify_cases(output)
    return m


def generate(output):
    verify_ready(output)
    experiment.generate(output)


def score(output):
    verify_ready(output)
    experiment.score(output)


def analyze(output):
    m = verify_ready(output)
    qs, keys = load(output/'public_questions.json'), load(output/'private_case_key.json')
    rows = []
    for aid,k in keys.items():
        s = load(output/'scores'/(aid+'.json'))
        g = load(output/'generations'/(aid+'.json'))
        rows.append(dict(aid=aid,**k,source=qs[k['question_key']]['sources'][0],quality=s['quality'],
          proxy=s['proxy_comparison']['signal']['status'],refusal=s['field_refusal']['refusal'],
          finish_reason=g['finish_reason']))
    table = [dict(condition=c,**metrics([r for r in rows if r['condition']==c])) for c in CONDITIONS]
    pairs = []
    for q in qs:
        a = next(r for r in rows if r['question_key']==q and r['condition']=='live_normal')
        for b in [r for r in rows if r['question_key']==q and r['condition']!='live_normal']:
            determined = a['quality']!='U' and b['quality']!='U'
            pairs.append(dict(question_key=q,source=a['source'],condition=b['condition'],
              normal_quality=a['quality'],stress_quality=b['quality'],normal_proxy=a['proxy'],stress_proxy=b['proxy'],
              worsened=int(determined and int(b['quality'])<int(a['quality'])),
              improved=int(determined and int(b['quality'])>int(a['quality'])),quality_u=int(not determined)))
    groups = [dict(source=src,condition=c,**metrics([r for r in rows if r['source']==src and r['condition']==c]))
              for src in sorted({r['source'] for r in rows}) for c in CONDITIONS]
    api = [load(p) for p in (output/'api_records').glob('*.json')]
    gens = [load(p) for p in (output/'generations').glob('*.json')]
    summary = dict(version=m['version'],interpretation=m['interpretation'],input_questions=len(qs),
      input_sources=len({v['sources'][0] for v in qs.values()}),fresh_generations=len(gens),
      qualified_cases=len(rows),excluded_before_answers=len(qs)*len(CONDITIONS)-len(rows),metrics=table,
      paired_worsened=sum(r['worsened'] for r in pairs),paired_improved=sum(r['improved'] for r in pairs),
      paired_quality_u=sum(r['quality_u'] for r in pairs),
      api_responses=len(api)+len(gens),failed_api_records=sum(not r.get('ok') for r in api),
      total_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in api+gens),
      actual_models=dict(Counter([r.get('model_returned','error') for r in api]+[r['actual_model'] for r in gens])),
      generation_finishes=dict(Counter(g['finish_reason'] for g in gens)),completed_at=utc(),
      assumptions='automatic_reference_quality; paired_questions_and_repeated_sources_not_independent_samples',
      detector_training_or_threshold_change=False,monitor_stress_plans_used=False,
      normal_scene_bad_answers_are_quality_errors_not_proxy_false_flags=True)
    for n,v in [('answer_scores.csv',rows),('signal_metrics.csv',table),('paired_quality.csv',pairs),('source_groups.csv',groups)]:
        write_csv(output/n,v)
    dump(output/'analysis_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','check_contexts','freeze_cases','generate','score','analyze'])
    ap.add_argument('--parent',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare(args.parent,args.output)
    else:
        globals()[args.action](args.output)
