"""V12.1 preparation: field aliases for ranking, corpus fallback intact tables.

V12's insufficient same-paper preparation remains frozen with zero API calls.
All source cell parsing and public calculation rules reuse the frozen V12 code.
Aliases change ranking presentation only, never returned evidence or gold values.
"""
import argparse
import copy
import re
from pathlib import Path
import rag_table_reasoning_stress as base
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load,dump,digest,sha,utc,write_csv
from rag_prospective_coverage_validation import verify_hashes

ALIASES={'训练集':['训练样本','训练数据','training'], '测试集':['测试样本','测试数据','test'],
         '收益均值':['平均收益','收益率'], '收益方差':['方差'], '企业估值':['估值'],
         'Gain(Y,X1)':['信息增益'], 'Gain(Y,X2)':['信息增益'], 'Gain(Y,X3)':['信息增益'], 'Gain(Y,X4)':['信息增益']}


def rank_view(text):
    text=base.visible(text).casefold()
    for a,b in [(r'\alpha','α'),(r'\beta','β'),(r'\gamma','γ'),(r'\bar','')]:text=text.replace(a,b)
    return text.replace('{','').replace('}','').replace('_','')


def select(spec,target,candidates):
    fields={rank_view(f['header_token']) for f in spec['fields']}|{rank_view(f['name']) for f in spec['fields']}
    aliases={rank_view(a) for f in spec['fields'] for a in ALIASES.get(f['name'],[])}
    ranked=[]
    for c in candidates:
        if c['source']==target['source'] and c['table_id']==target['table_id']:continue
        if len(c['text'])>base.CONFIG['distractor_max_characters']:continue
        text=rank_view(c['text']);direct=sum(t in text for t in fields);alias=sum(t in text for t in aliases)
        if not direct+alias:continue
        same=c['source']==target['source'];entities=sum(rank_view(e) in text for e in spec['pair'])
        rank=(int(same),direct,alias,entities,-len(c['text']),c['source'],-c['start'])
        ranked.append((rank,c))
    if not ranked:return None,dict(eligible_tables=0,rule='public_fields_aliases_same_paper_first_corpus_fallback')
    rank,c=max(ranked,key=lambda r:r[0])
    return copy.deepcopy(c),dict(eligible_tables=len(ranked),same_paper=bool(rank[0]),
        direct_field_hits=rank[1],alias_field_hits=rank[2],entity_hits=rank[3],selected_source=c['source'],
        selected_caption=c['caption'],selected_start=c['start'],public_alias_terms=sorted(aliases),
        rule='public_fields_aliases_same_paper_first_corpus_fallback_no_title_no_values')


def prepare(parent,failed,corpus,output):
    if (output/'protocol_lock.json').exists():base.verify(output);return
    prior=load(parent/'artifact_manifest.json');verify_hashes(prior['output_hashes']);verify_hashes(prior['code_and_input_hashes'])
    checkpoint=load(failed/'preparation_checkpoint_manifest.json');verify_hashes(checkpoint['output_hashes']);verify_hashes(checkpoint['code_and_input_hashes'])
    corpus_manifest=load(corpus/'source_manifest.json');qs=load(parent/'public_questions.json')
    indexed=[];texts={}
    for source,row in corpus_manifest.items():
        text=blocks_to_text(load(Path(row['mineru_path'])))
        if sha(text)!=row['text_sha256'] or digest(Path(row['mineru_path']))!=row['mineru_sha256']:raise ValueError('source changed')
        texts[source]=text;indexed+=base.tables(text,source)
    questions={};tasks={};refs={};banks={};screen=[];used={}
    for spec in base.specifications():
        bid=spec['bank_id'];source=qs[spec['seed_qid']]['sources'][0]
        targets=[t for t in indexed if t['source']==source and t['table_id']==spec['table_id']]
        if len(targets)!=1:raise ValueError('target locator not unique')
        table=targets[0];grid=base.parse_grid(spec,table);distractor,trace=select(spec,table,indexed)
        used[source]=corpus_manifest[source]
        if distractor:used[distractor['source']]=corpus_manifest[distractor['source']]
        variants={c:dict(retrieved=rows,evidence=base.evidence_projection(rows),context=base.source_context(rows),
            structurally_valid=bool(rows) and sum(len(r['text']) for r in rows)<=base.CONFIG['maximum_text_characters'],
            text_characters=sum(len(r['text']) for r in rows)) for c,rows in zip(base.CONDITIONS,[[table],[distractor,table] if distractor else []])}
        schema=dict(specification=spec,entities=[r['entity'] for r in grid],table_locator=dict(source=source,
            table_id=table['table_id'],caption=table['caption'],mineru_page=table['mineru_page']))
        banks[bid]=dict(source=source,public_schema=schema,target_table=table,private_source_grid=grid,
            variants=variants,distractor_selection_trace=trace)
        for kind in base.TYPES:
            qid=bid+'_'+kind;q,task=base.make_question(spec,table,kind)
            if q!=load(failed/'public_questions.json')[qid]:raise ValueError('question changed during evidence revision')
            questions[qid]=q;tasks[qid]=dict(bank_id=bid,**task);refs[qid]=base.make_plan(q,task,grid,table)
            screen.append(dict(question_key=qid,source=source,task_type=kind,candidate_status='pending',
                offline_source_and_calculation_valid=True,normal_structural=True,distractor_structural=variants[base.CONDITIONS[1]]['structurally_valid']))
    output.mkdir(parents=True,exist_ok=True)
    for n,v in [('public_questions.json',questions),('private_task_rules.json',tasks),('private_references.json',refs),
        ('table_banks.json',banks),('source_manifest.json',used),('offline_candidate_screening.json',screen)]:dump(output/n,v)
    write_csv(output/'offline_candidate_screening.csv',screen)
    old=load(failed/'protocol_lock.json')
    inputs=dict(old['input_hashes'])
    for p in [Path(__file__),Path(__file__).with_name('audit_rag_table_reasoning_revision.py'),
              Path(__file__).parent/'tests/test_rag_table_reasoning_revision.py',failed/'preparation_checkpoint_manifest.json',
              corpus/'source_manifest.json']:
        inputs[str(p.resolve())]=digest(p)
    files=[output/n for n in ['public_questions.json','private_task_rules.json','private_references.json','table_banks.json','source_manifest.json','offline_candidate_screening.json']]
    config=dict(old,version='table_reasoning_stress_v12_1',locked_at=utc(),previous=str(parent.resolve()),failed_preparation=str(failed.resolve()),
        source_corpus=str(corpus.resolve()),source_hashes={str(Path(v['mineru_path']).resolve()):v['mineru_sha256'] for v in used.values()},
        ranking_source_hashes={str(Path(v['mineru_path']).resolve()):v['mineru_sha256'] for v in corpus_manifest.values()},
        input_hashes=inputs,initial_hashes={str(p.resolve()):digest(p) for p in files},
        evidence_revision='case_math_and_subscript_presentation_aliases_for_ranking_only_same_paper_first_then_existing_corpus',
        target_papers=9,evidence_papers=len(used),ranking_corpus_papers=len(corpus_manifest),
        original_question_bank_unchanged=True,no_test_answers_seen_in_failed_preparation=True)
    dump(output/'protocol_lock.json',config)
    n=sum(b['variants'][base.CONDITIONS[1]]['structurally_valid'] for b in banks.values())
    if n*3<config['minimum_paired_questions'] or n<config['minimum_paired_sources']:
        dump(output/'offline_preflight_failure.json',dict(status='failed',structural_sources=n,structural_questions=n*3,generation_prohibited=True));raise ValueError('revised offline source gate failed')
    print('V12.1 prepared',len(questions),'new questions;',n,'paired table banks;',len(used),'evidence papers',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','failed','corpus','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();prepare(a.parent,a.failed,a.corpus,a.output)
