"""Complete retrieved passage tails within the existing raw character budget.

Deterministic postprocessing of public-query ranks. No normative facts or answer
are accepted. Source slices, original ranks, and complete contained tables remain
traceable. Unrepairable boundaries are retained and reported, never invented.
"""
import copy
import re

from rag_fresh_change_experiment import sha

CONFIG = dict(maximum_passage_characters=2000, maximum_total_characters=6000,
              maximum_passages=3, boundary_search_characters=256,
              boundaries="Chinese_sentence_end_or_newline_outside_paired_math",
              keep_ranks=True, preserve_complete_contained_tables=True)


def boundaries(text, start, end):
    math = [(m.start(), m.end()) for m in re.finditer(r"\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$", text, re.S)]
    return [i+1 for i in range(max(0,start),min(len(text),end))
            if text[i] in "。！？\n" and not any(a<=i<b for a,b in math)]


def repair_passage(passage, full_text):
    start,end=passage["start"],passage["end"]
    if not (0<=start<end<=len(full_text)) or full_text[start:end]!=passage["text"] or sha(passage["text"])!=passage["text_sha256"]:
        raise ValueError("input passage differs from original source")
    if end-start>CONFIG["maximum_passage_characters"]:raise ValueError("input passage exceeds budget")
    result=copy.deepcopy(passage);new_start,new_end=start,end
    trace=dict(original_start=start,original_end=end,original_text_sha256=passage["text_sha256"],
               status="unchanged_already_complete",added_tail_characters=0,removed_head_characters=0)
    safe=boundaries(full_text,start,min(len(full_text),end+CONFIG["boundary_search_characters"]))
    if end<len(full_text) and end not in safe:
        candidates=[p for p in safe if end<p<=end+CONFIG["boundary_search_characters"]]
        if not candidates:
            trace["status"]="unresolved_no_tail_boundary"
        else:
            proposed_end=candidates[0]
            minimum_start=max(start,proposed_end-CONFIG["maximum_passage_characters"])
            proposed_start=start
            if minimum_start>start:
                choices=[p for p in safe if minimum_start<=p<=start+CONFIG["boundary_search_characters"]]
                proposed_start=choices[0] if choices else None
            contained=[(start+m.start(),start+m.end()) for m in
                       re.finditer(r"\[TABLE_START\].*?\[/TABLE_END\]",passage["text"],re.S)]
            if proposed_start is None:
                trace["status"]="unresolved_no_head_budget_boundary"
            elif any(proposed_start>a or proposed_end<b for a,b in contained):
                trace["status"]="unresolved_would_cut_contained_table"
            else:
                new_start,new_end=proposed_start,proposed_end
                trace.update(status="repaired",added_tail_characters=new_end-end,
                             removed_head_characters=new_start-start)
    result.update(start=new_start,end=new_end,text=full_text[new_start:new_end],
                  text_sha256=sha(full_text[new_start:new_end]),boundary_repair=trace)
    return result


def repair_selected(passages, documents):
    if len(passages)>CONFIG["maximum_passages"]:raise ValueError("too many selected passages")
    repaired=[repair_passage(p,documents[p["source"]]) for p in passages]
    if sum(len(p["text"]) for p in repaired)>CONFIG["maximum_total_characters"]:
        raise ValueError("total passage budget exceeded")
    return repaired
