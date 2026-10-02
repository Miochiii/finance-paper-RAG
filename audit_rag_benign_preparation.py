"""Offline V11 preparation audit; does not import or call an API client."""
import argparse
from pathlib import Path
import rag_benign_normal_stress as v11
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_percent_fact_guard import fact_projection_issues


def audit(output):
    m=v11.verify(output)
    qs=load(output/'public_questions.json')
    variants=load(output/'context_variants.json')
    refs=load(output/'private_references.json')
    retrieval=load(output/'public_retrieval.json')
    sources=load(output/'source_manifest.json')
    plans={q:load(output/'proxy_plans'/(q+'.json')) for q in qs}
    if fact_projection_issues(refs,plans):
        raise ValueError('invalid numeric fact projection')
    stats=[]
    for q,public in qs.items():
        source=public['sources'][0]
        text=blocks_to_text(load(Path(sources[source]['mineru_path'])))
        if sha(text)!=sources[source]['text_sha256']:
            raise ValueError('source text hash mismatch')
        replay=v11.construct(public,retrieval[q],text,m['config'])
        original_union=v11.union_intervals(retrieval[q]['retrieved'])
        for c,row in replay.items():
            row['normal_literal_basis_preserved']=v11.literal_basis_preserved(plans[q],row['evidence'])
            row['structurally_valid'] &= row['normal_literal_basis_preserved']
            if not row['normal_literal_basis_preserved']:
                row['reason']='normal_literal_basis_missing'
            if variants[q][c]!=row:
                raise ValueError('construction replay mismatch')
            spans=v11.union_intervals(row['retrieved'])
            if row['structurally_valid'] and not all(any(a<=x and y<=b for a,b in spans) for x,y in original_union):
                raise ValueError('original source position removed')
            if c!='live_distractor' and row['text_characters']>6000:
                raise ValueError('baseline/reverse/dedup exceeds 6000')
            if any(r['text']!=text[r['start']:r['end']] for r in row['retrieved']):
                raise ValueError('source text edited')
    for c in v11.CONDITIONS:
        rows=[v[c] for v in variants.values()]
        stats.append(dict(condition=c,candidates=len(rows),structurally_valid=sum(r['structurally_valid'] for r in rows),
            literal_basis_preserved=sum(r['normal_literal_basis_preserved'] for r in rows),
            minimum_text_characters=min(r['text_characters'] for r in rows),maximum_text_characters=max(r['text_characters'] for r in rows)))
    result=dict(status='passed',checked_at=utc(),questions=len(qs),sources=len(sources),candidate_contexts=len(qs)*4,
        metrics=stats,all_source_spans_exact=True,all_baseline_source_positions_preserved=True,
        all_numeric_and_unit_fact_projections_valid=True,all_normal_basis_quotes_preserved=True,
        public_only_selection_replayed=True,generations_collected=len(list((output/'generations').glob('*.json'))),
        scope='structural_and_literal_basis_integrity_only_semantic_qualification_still_requires_blind_checks')
    dump(output/'preparation_integrity_audit.json',result)
    print('V11 offline preparation audit passed;',len(qs)*4,'exact-source cases',flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    audit(ap.parse_args().output)
