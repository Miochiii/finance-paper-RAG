# __init__.py
import os
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())  # 从包目录向上查找项目根目录的 .env
except ImportError:
    pass  # 未安装 python-dotenv 时降级为使用系统环境变量
from .classify_file import classifier
from .document_loader import load_document
from .chunk_splitter import dispatch_chunk
from .retriever import HybridRetriever, build_index_from_chunks
from .query_processor import QueryProcessor