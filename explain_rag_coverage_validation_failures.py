"""Classify validation U without changing frozen predictions or quality labels."""
import argparse
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,write_csv,utc


def explain(output):
    cases,keys,private=[load(output/n) for n in ('public_cases.json','public_proxy_plan_keys.json','private_case_key.json')]
    rows=[]
    for aid,case in cases.items():
        score,gen=load(output/'scores'/(aid+'.json')),load(output/'generations'/(aid+'.json'))
        plan=load(output/'proxy_plans'/(keys[aid]+'.json'))
        invalid=[]
        for n,raw in enumerate(score['comparisons'],1):
            for point in raw.get('point_results',[]):
                if point.get('status') in {'covered','partial','contradicted'}:
                    quote=auto.norm(point.get('answer_quote',''))
                    if len(quote)<4 or quote not in auto.norm(gen['answer']):
                        invalid.append(dict(round=n,index=point.get('index'),too_short=len(quote)<4,
                                            literal_not_found=quote not in auto.norm(gen['answer'])))
        if score['signal']['status']!='U':cause='determinate'
        elif not plan['basis_complete']:cause='temporary_basis_invalid'
        elif invalid:cause='answer_quote_validation'
        else:cause='classification_disagreement_or_other_schema'
        inconsistency=any(raw.get('answer_quality')=='2' and any(p.get('status')!='covered' for p in raw.get('point_results',[])) for raw in score['rounds'])
        rows.append(dict(aid=aid,**private[aid],quality=score['answer_quality'],signal_status=score['signal']['status'],
             signal_cause=cause,invalid_answer_quote_rows=len(invalid),short_quote_rows=sum(p['too_short'] for p in invalid),
             nonliteral_quote_rows=sum(p['literal_not_found'] for p in invalid),quality_total_inconsistent_with_points=int(inconsistency)))
    write_csv(output/'validation_failure_causes.csv',rows)
    value=dict(created_at=utc(),cases=len(rows),signal_cause_counts=dict(Counter(r['signal_cause'] for r in rows)),
          quality_u=sum(r['quality']=='U' for r in rows),quality_u_with_inconsistent_total=sum(r['quality']=='U' and r['quality_total_inconsistent_with_points'] for r in rows),
          frozen_scores_unchanged=True,diagnosis_only=True)
    dump(output/'failure_cause_summary.json',value);print(value)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    explain(ap.parse_args().output)
