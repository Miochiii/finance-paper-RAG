"""Candidate evidence from MinerU table/text blocks with raw provenance.

Do not treat chart reconstructions as exact numeric references. This selects
question-creation evidence, never passages for scoring or RAG generation.
"""
import re

from rag_core.mineru_loader import _block_to_text
from rag_fresh_change_experiment import sha

APPROXIMATE=re.compile(r"[~≈]|(?:约|大约|近似为)\s*[+-]?\d")


def block_regions(text,blocks):
    regions=[];cursor=0;image_index=[0]
    for index,block in enumerate(blocks):
        raw=_block_to_text(block,int(block.get("page_idx",0) or 0)+1,image_index)
        if not raw:continue
        start=text.find(raw,cursor)
        if start<0:raise ValueError("MinerU block differs from assembled original text")
        regions.append(dict(block_index=index,block_type=block.get("type"),page_idx=int(block.get("page_idx",0) or 0),
                            start=start,end=start+len(raw),text=raw))
        cursor=start+len(raw)
    return regions


def exact_cards(text,source,blocks,count=4):
    pool=[]
    for region in block_regions(text,blocks):
        raw=region["text"];kind=region["block_type"]
        digits=len(re.findall(r"\d+(?:\.\d+)?",raw))
        if kind not in {"table","text"} or APPROXIMATE.search(raw) or digits<3:continue
        if kind=="table" and 80<=len(raw)<=900:
            score=100+min(digits,30)+3*sum(w in raw for w in ("样本","参数","准确","收益","性能","描述","数据"))
        elif kind=="text" and 180<=len(raw)<=900 and any(w in raw for w in ("本文","样本","实验","数据","实证")):
            score=min(digits,20)
        else:continue
        pool.append((score,region))
    cards=[]
    for score,r in sorted(pool,key=lambda item:(-item[0],item[1]["start"]))[:count]:
        cards.append(dict(eid=f"E{len(cards)+1}",source=source,text=r["text"],start=r["start"],end=r["end"],
               text_sha256=sha(r["text"]),kind="exact_"+r["block_type"],selection_score=score,
               mineru_block_index=r["block_index"],mineru_block_type=r["block_type"],page_idx=r["page_idx"]))
    return cards
