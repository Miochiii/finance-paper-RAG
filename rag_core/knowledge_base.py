# knowledge_base.py
# 多源文件统一 chunk 存储：分类 → 加载 → 分块 → 聚合追加
import os
import json
from typing import List, Dict, Tuple

from rag_core.classify_file import classifier
from rag_core.document_loader import load_document
from rag_core.chunk_splitter import dispatch_chunk
from rag_core.config import KB_FILE

DEFAULT_STORE = KB_FILE


def _chunk_entries(chunks: List[str], source: str, source_type: str,
                   pdf_path: str = None, page_ranges: List[Tuple[int, int]] = None) -> List[Dict]:
    """把一批分块构造成知识库条目（含溯源元数据）。供 add_chunks 与增量入库共用。"""
    entries: List[Dict] = []
    for i, c in enumerate(chunks):
        if not c or not c.strip():
            continue
        entry = {
            "text": c,
            "source": source,
            "source_type": source_type,
        }
        if pdf_path:
            entry["pdf_path"] = pdf_path
        if page_ranges and i < len(page_ranges) and page_ranges[i][0] is not None:
            entry["page_start"] = page_ranges[i][0]
            entry["page_end"] = page_ranges[i][1]
        entries.append(entry)
    return entries


class KnowledgeBase:
    """
    聚合知识库：
    - process_file / add_files：处理新文件并暂存到内存
    - save()：合并写入 store_path（保留已有 + 追加新增，按 source+text 去重）
    - load() / to_texts_metadatas()：读取全部 chunk 供检索器使用
    """

    def __init__(self, store_path: str = None):
        self.store_path = store_path or DEFAULT_STORE
        self.pending: List[Dict] = []  # 本次新增，尚未落盘

    # ===== 添加文件 =====

    def process_file(self, file_path: str) -> int:
        """处理单个文件，返回本次新增 chunk 数（存到 pending）"""
        cfg = classifier.classify(file_path)
        raw_text = load_document(file_path, cfg["extractor"])
        if not raw_text.strip():
            print(f"  [警告] 提取为空，跳过: {os.path.basename(file_path)}")
            return 0

        chunk_size = cfg["chunk_size"]
        # DISCOURSE/HYBRID 需要整数 token 上限；TABULAR/CODE/NESTED 的 chunk_size
        # 是规则串（row_group_50 / func_aware / depth_aware），必须原样透传
        if cfg["structure"] in ("DISCOURSE", "HYBRID") and not isinstance(chunk_size, int):
            chunk_size = 800  # 兜底

        file_chunks = dispatch_chunk(
            text=raw_text,
            structure=cfg["structure"],
            chunk_size=chunk_size,
            overlap_tokens=50,
        )

        source_name = os.path.basename(file_path)
        added = 0
        for c in file_chunks:
            if c.strip():
                self.pending.append({
                    "text": c,
                    "source": source_name,
                    "source_type": cfg["structure"],
                })
                added += 1
        print(f"  {source_name}: +{added} 块 ({cfg['structure']})")
        return added

    def add_files(self, file_paths: List[str]) -> int:
        """批量处理文件，返回累计新增 chunk 数"""
        total = 0
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"  [跳过] 文件不存在: {fp}")
                continue
            try:
                total += self.process_file(fp)
            except Exception as e:
                print(f"  [失败] {os.path.basename(fp)}: {e}")
        return total

    # ===== 添加纯文本段 =====

    def add_text(self, text: str, source: str = None, structure: str = "DISCOURSE", chunk_size: int = 800, overlap_tokens: int = 50) -> int:
        """直接添加一段纯文本到知识库（不经过文件分类/加载）。
        适用于手动补充的笔记、摘录、片段等。
        - text: 文本内容
        - source: 来源名称（用于溯源与去重），默认自动生成
        - structure: 分块策略（DISCOURSE 文本 / HYBRID 带表格标记）
        """
        if not text or not text.strip():
            print("  [警告] 文本为空，跳过")
            return 0
        if not source:
            import datetime
            source = "手动文段_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_chunks = dispatch_chunk(
            text=text.strip(),
            structure=structure,
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
        )
        added = 0
        for c in file_chunks:
            if c.strip():
                self.pending.append({
                    "text": c,
                    "source": source,
                    "source_type": structure,
                })
                added += 1
        print(f"  {source}: +{added} 块 ({structure})")
        return added

    def add_chunks(self, chunks: List[str], source: str, source_type: str,
                   pdf_path: str = None, page_ranges: List[Tuple[int, int]] = None) -> int:
        """直接追加一批已分块文本（跳过分类/加载/分块），返回实际加入条数。
        供 build_kb 等外部管线在完成 HMM/规则分块后写入。
        pdf_path / page_ranges 为溯源元数据：page_ranges 与 chunks 同长，
        元素为 (起始页, 结束页) 或 (None, None)。"""
        entries = _chunk_entries(chunks, source, source_type, pdf_path, page_ranges)
        return self.add_entries(entries)

    def add_entries(self, entries: List[Dict]) -> int:
        """直接追加已构造好的条目列表（供增量入库等外部管线复用同一份元数据）。"""
        self.pending.extend(entries)
        return len(entries)

    # ===== 持久化 =====

    def save(self):
        """合并写入：已有 + 新增，按 (source, text) 去重。可重复调用。"""
        try:
            existing = self._load_all()
        except ValueError as e:
            # 损坏文件先备份，绝不静默覆盖导致知识库丢失
            backup = self.store_path + ".corrupt"
            try:
                os.replace(self.store_path, backup)
                print(f"[警告] {e}，已备份到 {backup}，将重建知识库")
            except OSError:
                print(f"[警告] {e}，且无法备份，将以空库重建")
            existing = []
        seen = {(c["source"], c["text"]): idx for idx, c in enumerate(existing)}
        for c in self.pending:
            key = (c["source"], c["text"])
            if key in seen:
                # 同块已存在：合并新增的溯源元数据（页码/路径），不重复追加
                target = existing[seen[key]]
                for k, v in c.items():
                    if k in ("text", "source"):
                        continue
                    if v not in (None, "", []):
                        target[k] = v
            else:
                existing.append(c)
                seen[key] = len(existing) - 1
        self.pending = []

        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"  已保存 {len(existing)} 块 → {self.store_path}")

    def load(self) -> List[Dict]:
        """读取 store_path 全部 chunk"""
        return self._load_all()

    def _load_all(self) -> List[Dict]:
        if not os.path.exists(self.store_path):
            return []
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("内容不是 JSON 数组")
            return data
        except Exception as e:
            raise ValueError(f"知识库文件损坏: {self.store_path} ({e})")

    # ===== 供检索器使用 =====

    def to_texts_metadatas(self) -> Tuple[List[str], List[Dict]]:
        """返回 (texts, metadatas)，metadatas 带 source 用于引文溯源"""
        chunks = self.load()
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", ""),
                "source_type": c.get("source_type", ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "pdf_path": c.get("pdf_path"),
            }
            for c in chunks
        ]
        return texts, metadatas

    # ===== 便捷查询 =====

    def stats(self) -> str:
        """返回知识库统计信息"""
        chunks = self.load()
        if not chunks:
            return "知识库为空"
        by_source = {}
        for c in chunks:
            by_source[c.get("source", "未知")] = (
                by_source.get(c.get("source", "未知"), 0) + 1
            )
        lines = [f"总块数: {len(chunks)}"]
        for src, cnt in sorted(by_source.items()):
            lines.append(f"  {src}: {cnt} 块")
        return "\n".join(lines)
