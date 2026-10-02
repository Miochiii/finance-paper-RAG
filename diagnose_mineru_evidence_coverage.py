"""Local question-only retrieval diagnostic; literal quotes are offline probes.

No API calls. Visible source metadata determines the scope. Normative quotes
are never used to select or rerank passages, only to report literal availability.
This is not semantic answer validation or a production retrieval benchmark.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,digest,sha,utc,write_csv
from rag_core.mineru_loader import blocks_to_text
from rag_subtle_fault_pilot import LocalReranker,sources


def make_chunks(text,filename,size=2000,stride=1500):
    if size<stride or stride<=0:raise ValueError('invalid overlapping window')
    chunks=[]
    for start in range(0,len(text),stride):
        end=min(start+size,len(text));part=text[start:end]
        if len(auto.norm(part))>=80:chunks.append(dict(source=filename,start=start,end=end,text=part,text_sha256=sha(part)))
        if end==len(text):break
    return chunks


def retrieve(question,visible_sources,chunks,vectorizer,matrix,model,candidates=20,top_k=3):
    # Explicit interface: no qid, gold source, normative fact or answer enters ranking.
    indexes=[i for i,row in enumerate(chunks) if row['source'] in visible_sources]
    if not indexes:return []
    query=vectorizer.transform([question])
    lexical=np.asarray((matrix[indexes]@query.T).toarray()).reshape(-1)
    order=sorted(range(len(indexes)),key=lambda p:(-lexical[p],chunks[indexes[p]]['source'],chunks[indexes[p]]['start']))[:candidates]
    shortlist=[chunks[indexes[p]] for p in order]
    scores=model.score([(question,c['text']) for c in shortlist],4096)
    ranked=sorted(zip(shortlist,scores),key=lambda x:(-x[1]['logit'],x[0]['source'],x[0]['start']))
    return [dict(**chunk,rerank=score) for chunk,score in ranked[:top_k]]


def literal_coverage(points,texts):
    normalized=[auto.norm(text) for text in texts]
    return [any(auto.norm(p['quote']) in text for text in normalized) for p in points]


def run(validation,prior,output):
    output.mkdir(parents=True,exist_ok=True)
    selected=load(validation/'private_screen_inputs.json')
    source_manifest=load(prior/'source_manifest.json')
    documents,chunks,manifest=[],[],{}
    # Complete frozen document index; actual visible metadata controls each query's scope.
    for filename,row in sorted(source_manifest.items()):
        path=Path(row['mineru_path'])
        if digest(path)!=row['mineru_sha256']:raise ValueError('MinerU original changed')
        text=blocks_to_text(load(path));documents.append(dict(source=filename,text=text))
        chunks.extend(make_chunks(text,filename))
        manifest[str(path)]=row['mineru_sha256']
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,3),max_features=60000,lowercase=True,dtype=np.float32)
    matrix=vectorizer.fit_transform([c['text'] for c in chunks])
    cfg=load(validation/'protocol_lock.json')['config']
    model=LocalReranker(cfg['reranker_path'])
    rows,records=[],{}
    for qid,row in selected.items():
        scope=sources(row['context'])
        found=retrieve(row['question'],scope,chunks,vectorizer,matrix,model)
        old=literal_coverage(row['points'],[row['context']])
        fresh=literal_coverage(row['points'],[c['text'] for c in found])
        full=literal_coverage(row['points'],[d['text'] for d in documents if d['source'] in scope])
        states=load(validation/'construction'/(qid+'.json'))['baseline_statuses']
        rows.append(dict(qid=qid,necessary_points=len(old),cache_literal_matches=sum(old),retrieved_literal_matches=sum(fresh),
             whole_visible_mineru_matches=sum(full),cache_all_quotes=int(all(old)),retrieved_all_quotes=int(all(fresh)),
             whole_visible_mineru_all_quotes=int(all(full)),visible_sources=len(scope),indexed_visible_sources=sum(d['source'] in scope for d in documents),
             retrieved_passages=len(found),retrieved_characters=sum(len(c['text']) for c in found),
             baseline_semantic_complete=int(all(s=='supported' for run in states for s in run))))
        records[qid]=dict(question=row['question'],actual_visible_sources=sorted(scope),retrieved=found,
              literal_matches=dict(cache=old,retrieved=fresh,whole_visible_source=full),
              offline_normative_points=row['points'])
        print(qid,'literal',sum(old),'->',sum(fresh),'/',len(old),'full',sum(full),flush=True)
    del model
    write_csv(output/'literal_evidence_coverage.csv',rows)
    dump(output/'local_retrieval_records.json',records)
    value=dict(questions=len(rows),documents=len(documents),chunks=len(chunks),api_calls=0,
         cache_all_quotes=sum(r['cache_all_quotes'] for r in rows),retrieved_all_quotes=sum(r['retrieved_all_quotes'] for r in rows),
         full_visible_mineru_all_quotes=sum(r['whole_visible_mineru_all_quotes'] for r in rows),
         total_points=sum(r['necessary_points'] for r in rows),cache_quote_matches=sum(r['cache_literal_matches'] for r in rows),
         retrieved_quote_matches=sum(r['retrieved_literal_matches'] for r in rows),whole_visible_source_quote_matches=sum(r['whole_visible_mineru_matches'] for r in rows),
         improved_questions=sum(r['retrieved_literal_matches']>r['cache_literal_matches'] for r in rows),
         worsened_questions=sum(r['retrieved_literal_matches']<r['cache_literal_matches'] for r in rows),
         original_semantically_complete=sum(r['baseline_semantic_complete'] for r in rows),
         interpretation='literal_quote_availability_only_not_semantic_coverage_or_answer_accuracy')
    dump(output/'analysis_summary.json',value)
    dump(output/'manifest.json',dict(created_at=utc(),input_hashes={str(p.resolve()):digest(p) for p in
         [Path(__file__),validation/'private_screen_inputs.json',prior/'source_manifest.json']},
         mineru_hashes=manifest,config=dict(window_chars=2000,stride_chars=1500,lexical_candidates=20,rerank_max_tokens=4096,top_k=3,
         scope='actual_cached_visible_sources_no_gold_source_filter'),ranker_inputs_exclude_normative_facts=True))
    audit_records(records,documents)
    dump(output/'integrity_audit.json',dict(status='passed',checked_at=utc(),original_source_hashes_checked=len(manifest),
         retrieved_spans_match_original_text=True,metadata_scope_checked=True,no_api_calls=True,
         normative_quotes_used_only_after_ranking=True))
    lines=['# MinerU 本地检索诊断：逐字证据可用性','',
      '本诊断没有调用 API，没有改动本轮验证方法或原服务。按问题和原缓存实际可见来源筛选全文段落，字符 TF-IDF 取前20个候选，再用本地 BGE 4096 token 重排，取前3段。标准事实/摘录不参与检索。','',
      '**下表仅比较冻结原文摘录是否逐字出现在文本中。缺少同一摘录不等于缺少语义支持；摘录存在也不保证回答完整。本结果不能替代自动语义核查。**','',
      '|题号|必要点|原缓存摘录数|检索摘录数|可见来源全文摘录数|原缓存语义完整|','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ('qid','necessary_points','cache_literal_matches','retrieved_literal_matches','whole_visible_mineru_matches','baseline_semantic_complete'))+'|')
    lines += ['',f"共 {value['documents']} 篇原文、{value['chunks']} 个窗口；{value['questions']} 题、{value['total_points']} 个必要点。逐字摘录命中：原缓存 {value['cache_quote_matches']}，问题检索 {value['retrieved_quote_matches']}，实际可见来源全文 {value['whole_visible_source_quote_matches']}。",'',
      f"相较原缓存，{value['improved_questions']} 题逐字命中增加，{value['worsened_questions']} 题减少。不能仅根据这些数字宣布全文检索更好；需要独立固定检索配置，再对语义支持和新回答进行验证。",'',
      '下一步区分三类情况：全文有但检索未取到，说明候选召回或重排仍需改进；实际可见来源全文也无相应原文，需检查来源范围；上下文有事实但必要点范围有争议，需先审查评价依据。所有修订作为后续开发，不能替换已完成验证的数据。','']
    (output/'mineru_diagnostic.md').write_text('\n'.join(lines),encoding='utf-8')
    print(value)


def audit_records(records,documents):
    texts={d['source']:d['text'] for d in documents}
    for row in records.values():
        for span in row['retrieved']:
            if span['source'] not in row['actual_visible_sources'] or texts[span['source']][span['start']:span['end']]!=span['text']:
                raise ValueError('retrieval scope or exact span mismatch')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('validation','prior','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.validation,a.prior,a.output)
