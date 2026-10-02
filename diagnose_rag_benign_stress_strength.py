"""Post hoc context manipulation diagnostics; never filters answers or scores."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,utc,write_csv
from rag_objective_slots import exact_source_quote
from rag_benign_normal_stress import verify,CONDITIONS


def diagnose(output):
    verify(output)
    qs=load(output/'public_questions.json')
    variants=load(output/'context_variants.json')
    rows=[]
    for qid,q in qs.items():
        normal=load(output/'proxy_plans'/(qid+'.json'))
        bank=variants[qid]
        for condition in CONDITIONS:
            row=bank[condition]
            context=row['context']
            ends=[exact_source_quote(s['quote'],context)['end']/len(context) for s in normal['slots']]
            baseline=bank['live_normal']['text_characters']
            rows.append(dict(question_key=qid,condition=condition,
                source=q['sources'][0],text_characters=row['text_characters'],
                baseline_text_characters=baseline,passages=len(row['retrieved']),
                context_changed=int(context!=bank['live_normal']['context']),
                removed_duplicate_characters=baseline-bank['live_dedup']['text_characters'],
                latest_required_quote_end_fraction=max(ends),
                earliest_required_quote_end_fraction=min(ends)))
    distractors=[b['live_distractor']['retrieved'][0] for b in variants.values()]
    table=[dict(condition=c,contexts=sum(r['condition']==c for r in rows),
         contexts_changed=sum(r['context_changed'] for r in rows if r['condition']==c),
         mean_text_characters=mean(r['text_characters'] for r in rows if r['condition']==c),
         mean_latest_required_quote_end_fraction=mean(r['latest_required_quote_end_fraction'] for r in rows if r['condition']==c))
         for c in CONDITIONS]
    cover_hints=['硕士学位论文','专业学位硕士','学位类别','论文答辩日期','论文题目']
    summary=dict(diagnosed_at=utc(),scope='posthoc_strength_description_no_selection_or_threshold_tuning',
       questions=len(qs),metrics=table,distractor_kinds=dict(Counter(r['kind'] for r in distractors)),
       distractors_with_cover_metadata=sum(any(h in r['text'] for h in cover_hints) for r in distractors),
       distractor_characters=dict(minimum=min(len(r['text']) for r in distractors),
          maximum=max(len(r['text']) for r in distractors),mean=mean(len(r['text']) for r in distractors)),
       questions_with_actual_overlap_removed=sum(b['live_dedup']['text_characters']<b['live_normal']['text_characters'] for b in variants.values()),
       mean_removed_duplicate_characters=mean(b['live_normal']['text_characters']-b['live_dedup']['text_characters'] for b in variants.values()),
       all_baseline_facts_preserved=True,
       limitation='public_full_question_bigram_ranking_includes_paper_title_many_distractors_are_general_background_only_one_other_intact_table',
       answers_used=False,scoring_used=False)
    write_csv(output/'stress_strength.csv',rows)
    dump(output/'stress_strength_summary.json',summary)
    print('Stress strength:',summary['distractor_kinds'],'actual dedup',summary['questions_with_actual_overlap_removed'],
          'cover metadata',summary['distractors_with_cover_metadata'],flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    diagnose(ap.parse_args().output)
