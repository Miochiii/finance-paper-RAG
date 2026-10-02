"""Conservative numeric reading of the third result field; offline supplement.

This does not change reference/proxy grades. Unreadable layouts remain U rather
than selecting a number using the expected answer. The parser is frozen before
V12.1 test answer generation and does not see gold when choosing a result.
"""
import argparse
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from collections import Counter
from rag_fresh_change_experiment import load,dump,utc,write_csv

FINAL_NUMBER=re.compile(r'(?<![\d.])(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:\(?\s*(?P<unit>倍|百分点|%)\s*\)?)?\s*[。．.,，;；]*$')


def result_literal(answer,kind):
    normalized=unicodedata.normalize('NFKC',answer).replace('−','-').replace('–','-')
    numbered=[];marked=[]
    for line in normalized.splitlines():
        s=line.strip().replace('**','').replace('`','').replace('$','')
        s=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',s)
        s=s.replace(r'\(', '').replace(r'\)', '')
        s=re.sub(r'^[-*]\s+','',s)
        if re.match(r'^(?:3\s*[.)、]|\(3\))',s):
            numbered.append(s)
        markers=['差值','A减B','A-B'] if kind=='difference' else ['比值','A除以B','A/B','A÷B']
        if any(m in s for m in markers):marked.append(s)
    selected=numbered or marked
    if len(selected)!=1:return dict(status='U',reason='result_field_not_unique',literal=None,unit=None,quote=None)
    s=selected[0]
    body=re.sub(r'^(?:3\s*[.)、]|\(3\))\s*','',s)
    match=FINAL_NUMBER.search(body)
    if not match or (match.start()==0 and not numbered):return dict(status='U',reason='result_requires_explicit_field_or_separator',literal=None,unit=None,quote=s)
    if match.start()>0 and body[match.start()-1] in 'eE':return dict(status='U',reason='scientific_notation_unsupported',literal=None,unit=None,quote=s)
    return dict(status='readable',reason='last_scalar_at_end_of_unique_result_field',literal=match['number'],unit=match['unit'] or '',quote=s)


def diagnose(output):
    cases=load(output/'public_cases.json');keys=load(output/'private_case_key.json');tasks=load(output/'private_task_rules.json');refs=load(output/'private_references.json')
    rows=[]
    for aid in cases:
        key=keys[aid];qid=key['question_key'];kind=tasks[qid]['rule']['kind']
        if kind not in ['difference','ratio']:continue
        answer=load(output/'generations'/(aid+'.json'))['answer']
        parsed=result_literal(answer,kind)
        expected=refs[qid]['facts'][2]
        matched=parsed['status']=='readable' and Decimal(parsed['literal'])==Decimal(expected['expected'])
        explicit_unit_wrong=parsed['status']=='readable' and bool(parsed['unit']) and parsed['unit']!=expected['unit']
        rows.append(dict(aid=aid,**key,task_type=kind,read_status=parsed['status'],reason=parsed['reason'],
            numeric_matches=int(matched) if parsed['status']=='readable' else 'U',
            explicit_unit_conflict=int(explicit_unit_wrong) if parsed['status']=='readable' else 'U',
            primary_quality=load(output/'scores'/(aid+'.json'))['quality']))
    summary=dict(status='completed',checked_at=utc(),calculated_answers=len(rows),
       readable=sum(r['read_status']=='readable' for r in rows),unreadable_U=sum(r['read_status']=='U' for r in rows),
       numeric_matches=sum(r['numeric_matches']==1 for r in rows),numeric_mismatches=sum(r['numeric_matches']==0 for r in rows),
       explicit_unit_conflicts=sum(r['explicit_unit_conflict']==1 for r in rows),
       primary_complete_but_numeric_mismatch=sum(r['primary_quality']=='2' and r['numeric_matches']==0 for r in rows),
       scopes=dict(Counter(r['task_type'] for r in rows)),
       no_gold_used_to_select_answer_number=True,primary_grades_changed=False,
       limitation='checks_only_explicit_third_field_numeric_literal_and_obvious_explicit_unit_conflicts_not_all_answer_semantics')
    write_csv(output/'calculation_literal_diagnostic.csv',rows);dump(output/'calculation_literal_diagnostic.json',summary)
    print('Calculation literals:',summary,flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    diagnose(ap.parse_args().output)
