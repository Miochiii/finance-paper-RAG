"""Public scope/field retrieval with intact tables and exact raw provenance.

No reference values, reference quotes, answer, quality or condition are accepted.
Normalization is for ranking only. Returned passages are original source slices.
"""
import re
import unicodedata

import numpy as np

from diagnose_mineru_evidence_coverage import make_chunks
from rag_fresh_change_experiment import sha

CONFIG = dict(window_chars=2000, stride_chars=1500, table_margin=250,
              lexical_per_query=12, top_k=3, maximum_characters=6000,
              rerank_max_tokens=4096, overlap_limit=.65)
PUBLIC_FIELDS = {"question", "requested_slots", "sources"}
SLOT_FIELDS = {"index", "label", "query_quote", "type"}


def surface_map(text):
    """Small explicit presentation map; never rewrite numbers or math operators."""
    out, positions = [], []
    i = 0
    replacements = {r"\varepsilon": "ε", r"\epsilon": "ϵ", r"\_": "_"}
    while i < len(text):
        found = next((k for k in replacements if text.startswith(k, i) and
                      (k == r"\_" or i+len(k) == len(text) or not text[i+len(k)].isalpha())), None)
        if found:
            out.append(replacements[found]); positions.append((i, i+len(found)))
            i += len(found); continue
        # NFKC collapses the two Greek epsilon glyphs; keep that distinction.
        visible = text[i] if text[i] in {"ε", "ϵ"} else unicodedata.normalize("NFKC", text[i])
        for char in visible:
            if not char.isspace() and char != "$":
                out.append(char); positions.append((i, i+1))
        i += 1
    return "".join(out), positions


def source_surface_anchor(value, quote):
    normalized, offsets = surface_map(quote)
    needle, _ = surface_map(value)
    start = normalized.find(needle) if needle else -1
    if start < 0: return None
    left, right = offsets[start][0], offsets[start+len(needle)-1][1]
    return dict(requested=value, literal=quote[left:right], start=left, end=right,
                mapping="nfkc_whitespace_math_delimiters_escaped_underscore_epsilon_only")


def public_query(row):
    if set(row) != PUBLIC_FIELDS: raise ValueError("public query contains non-public fields")
    slots = row["requested_slots"]
    if any(set(s) != SLOT_FIELDS for s in slots): raise ValueError("slot projection contains private fields")
    # Deterministic removal of the paper title and program's answer-format template.
    text = row["question"]
    scope = text.split("针对", 1)[-1].split("请仅逐项给出以下字段", 1)[0].strip("。，, ")
    labels = [s["label"] for s in slots]
    full = scope+"；"+"；".join(labels)
    # Each field remains paired with its public object/period/table scope.
    queries = list(dict.fromkeys([full]+[scope+"；"+label for label in labels]))
    return dict(scope=scope, labels=labels, queries=queries,
                table_ids=table_ids(scope), sources=list(row["sources"]))


def table_ids(text):
    # Table 5-1 and Table 5.1 denote the same public table locator here.
    return sorted(set(f"{m[0]}-{m[1]}" for m in
                      re.findall(r"表\s*(\d+)\s*[-－–—.]\s*(\d+)", text)))


def field_chunks(text, source):
    rows = [dict(**r, kind="window", table_ids=[]) for r in make_chunks(text, source)]
    for match in re.finditer(r"\[TABLE_START\].*?\[/TABLE_END\]", text, re.S):
        head = match.group().splitlines()[1] if len(match.group().splitlines())>1 else ""
        ids = table_ids(head)
        length = match.end()-match.start()
        if length <= CONFIG["window_chars"]:
            margin = min(CONFIG["table_margin"], (CONFIG["window_chars"]-length)//2)
            starts_ends = [(max(0, match.start()-margin), min(len(text), match.end()+margin))]
        else:
            starts_ends = [(s, min(s+CONFIG["window_chars"], match.end()))
                          for s in range(match.start(), match.end(), CONFIG["stride_chars"])]
        for start, end in starts_ends:
            part = text[start:end]
            rows.append(dict(source=source, start=start, end=end, text=part,
                             text_sha256=sha(part), kind="table", table_ids=ids))
    unique = {}
    for r in rows:
        key = (source, r["start"], r["end"])
        if key not in unique or r["kind"] == "table": unique[key] = r
    return list(unique.values())


def overlap(a, b):
    if a["source"] != b["source"]: return 0
    intersection = max(0, min(a["end"], b["end"])-max(a["start"], b["start"]))
    return intersection/min(a["end"]-a["start"], b["end"]-b["start"])


def retrieve_fields(row, chunks, vectorizer, matrix, model):
    query = public_query(row)
    indexes = [i for i, c in enumerate(chunks) if c["source"] in query["sources"]]
    if not indexes: return [], query
    lexical = np.asarray((matrix[indexes] @ vectorizer.transform(
        [surface_map(q)[0].lower() for q in query["queries"]]).T).toarray())
    selected = set()
    for col in range(lexical.shape[1]):
        order = sorted(range(len(indexes)), key=lambda p:(-lexical[p,col], indexes[p]))
        selected.update(indexes[p] for p in order[:CONFIG["lexical_per_query"]])
    # Only explicit public table locators can bypass lexical candidate pruning.
    selected.update(i for i in indexes if chunks[i]["kind"] == "table" and
                    set(chunks[i]["table_ids"]) & set(query["table_ids"]))
    shortlist = [chunks[i] for i in sorted(selected)]
    scores = model.score([(query["queries"][0], c["text"]) for c in shortlist], CONFIG["rerank_max_tokens"])
    ranked = []
    for c, score in zip(shortlist, scores):
        exact_table = bool(set(c["table_ids"]) & set(query["table_ids"]))
        visible, _ = surface_map(c["text"])
        label_hits = sum(surface_map(label)[0].lower() in visible.lower() for label in query["labels"])
        ranked.append(dict(**c, rerank=score, exact_public_table=exact_table, public_label_hits=label_hits))
    # Exact named tables are considered first. Field hits break ties between continuations.
    ranked.sort(key=lambda c:(-int(c["exact_public_table"]),
                             -c["public_label_hits"] if c["exact_public_table"] else 0,
                             -c["rerank"]["logit"], c["source"], c["start"]))
    found = []
    for c in ranked:
        if any(overlap(c, old) > CONFIG["overlap_limit"] for old in found): continue
        if sum(len(old["text"]) for old in found)+len(c["text"]) > CONFIG["maximum_characters"]: continue
        found.append(c)
        if len(found) == CONFIG["top_k"]: break
    return found, dict(**query, candidate_count=len(shortlist), ranking_inputs="public_question_fields_sources_only")
