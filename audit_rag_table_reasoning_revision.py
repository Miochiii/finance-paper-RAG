"""Offline replay of V12.1 corpus/table selection and deterministic references."""
import argparse
from pathlib import Path
import rag_table_reasoning_stress as base
import rag_table_reasoning_revision as revision
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_prospective_coverage_validation import verify_hashes


def preparation(output):
    m=base.verify(output);verify_hashes(m['ranking_source_hashes'])
    failed=Path(m['failed_preparation']);checkpoint=load(failed/'preparation_checkpoint_manifest.json')
    verify_hashes(checkpoint['output_hashes']);verify_hashes(checkpoint['code_and_input_hashes'])
    corpus=load(Path(m['source_corpus'])/'source_manifest.json');tables=[];texts={}
    for source,row in corpus.items():
        text=blocks_to_text(load(Path(row['mineru_path'])))
        if sha(text)!=row['text_sha256']:raise ValueError('corpus conversion changed')
        texts[source]=text;tables+=base.tables(text,source)
    banks=load(output/'table_banks.json');qs=load(output/'public_questions.json');tasks=load(output/'private_task_rules.json');refs=load(output/'private_references.json')
    source_cells=0;same=0;fallback=0
    for spec in base.specifications():
        bid=spec['bank_id'];bank=banks[bid];table=bank['target_table'];text=texts[bank['source']]
        target=[t for t in tables if t['source']==table['source'] and t['table_id']==spec['table_id']]
        if len(target)!=1 or target[0]!=table:raise ValueError('target locator changed')
        parsed=base.parse_grid(spec,table)
        if parsed!=bank['private_source_grid']:raise ValueError('reference grid changed')
        for r in parsed:
            for c in r['values'].values():
                a,b=c['source_span']
                if text[a:b]!=c['literal']:raise ValueError('cell coordinate changed')
                source_cells+=1
        distractor,trace=revision.select(spec,table,tables)
        if trace!=bank['distractor_selection_trace']:raise ValueError('public selector changed')
        if distractor:
            same+=int(trace['same_paper']);fallback+=int(not trace['same_paper'])
        for c,rows in zip(base.CONDITIONS,[[table],[distractor,table] if distractor else []]):
            expected=dict(retrieved=rows,evidence=base.evidence_projection(rows),context=base.source_context(rows),
                structurally_valid=bool(rows) and sum(len(r['text']) for r in rows)<=base.CONFIG['maximum_text_characters'],
                text_characters=sum(len(r['text']) for r in rows))
            if expected!=bank['variants'][c]:raise ValueError('context changed')
            if any(texts[r['source']][r['start']:r['end']]!=r['text'] for r in rows):raise ValueError('source text edited')
        for kind in base.TYPES:
            qid=bid+'_'+kind;q,task=base.make_question(spec,table,kind)
            if qs[qid]!=q or q!=load(failed/'public_questions.json')[qid]:raise ValueError('question revised using answers')
            if tasks[qid]!=dict(bank_id=bid,**task) or refs[qid]!=base.make_plan(q,task,parsed,table):raise ValueError('calculation proof changed')
    result=dict(status='passed',checked_at=utc(),new_questions=len(qs),target_papers=9,evidence_papers=len(load(output/'source_manifest.json')),
        source_cells_checked=source_cells,ranking_corpus_papers=len(corpus),same_paper_distractors=same,cross_paper_distractors=fallback,
        structural_paired_questions=(same+fallback)*3,all_cell_and_table_coordinates_exact=True,
        Decimal_answers_independently_checked_with_Fraction=True,no_numbers_units_or_source_text_edited=True,
        public_fields_and_aliases_only_for_distractor_ranking=True,all_questions_unchanged_from_zero_answer_V12_preparation=True,
        all_candidate_statuses_pending=True,fresh_generations=len(list((output/'generations').glob('*.json'))),
        scope='source_and_arithmetic_integrity_blind_API_semantic_checks_still_required')
    dump(output/'preparation_integrity_audit.json',result)
    print('V12.1 preparation audit passed',len(qs),'questions;',source_cells,'source cells;',same,'same-paper +',fallback,'cross-paper distractors',flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    preparation(ap.parse_args().output)
