# query_processor.py
# 检索前 Query 预处理：改写/扩展、意图识别、关键词提取
import os
import json
import re
from typing import List, Dict, Optional

# ---------- 配置 ----------

DEEPSEEK_API_KEY = os.getenv("deepseek_api", "")  # 从环境变量读取，勿硬编码

_INTENT_PROMPT = """分析以下用户问题，返回 JSON：

{
  "intent": "factual" | "summary" | "mixed",
  "keywords": ["关键词1", "关键词2", ...],
  "sub_queries": ["改写/扩展后的子问题1", "子问题2", ...]
}

规则：
- factual：询问具体知识、定义、定理、公式、数值
- summary：要求概括、总结、综述
- mixed：同时包含事实查询和概括需求
- keywords：提取 3~8 个核心术语，保留专有名词原样（人名、定理名、公式名）
- sub_queries：如果问题简短模糊，拆成 2~3 个更具体的子问题；如果已经清晰，返回原问题

只输出 JSON，不要任何解释文字。"""


def _call_deepseek(prompt: str, user_msg: str, timeout: int = 60, retries: int = 1) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                stream=False,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                import time
                time.sleep(2)
    raise last_err


def _parse_json(text: str) -> Dict:
    """容错解析 LLM 返回的 JSON（可能被 markdown 代码块包裹）"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1)
    return json.loads(text)


# ---------- Query 处理器 ----------

class QueryProcessor:
    """
    查询预处理流水线：
    1. 意图识别（factual / summary / mixed）
    2. 关键词提取（辅助 BM25）
    3. 查询改写/扩展（多子问题召回）
    """

    def __init__(self):
        self._last_result: Optional[Dict] = None

    def process(self, question: str) -> Dict:
        """
        处理用户问题，返回：
        {
            "intent": "factual" | "summary" | "mixed",
            "keywords": [str, ...],
            "sub_queries": [str, ...],
            "expanded_query": str  # 拼接所有子问题，用于向量检索
        }
        """
        try:
            raw = _call_deepseek(_INTENT_PROMPT, question)
            result = _parse_json(raw)
        except Exception:
            # LLM 调用失败/超时/返回非法 JSON → 保守回退，不抛异常
            result = {
                "intent": "factual",
                "keywords": [],
                "sub_queries": [question],
            }

        result.setdefault("intent", "factual")
        # LLM 返回的 keywords/sub_queries 可能不是 list 或含非字符串 → 清洗，防下游 join 崩溃
        kw = result.get("keywords")
        result["keywords"] = (
            [str(k).strip() for k in kw if str(k).strip()] if isinstance(kw, list) else []
        )
        sq = result.get("sub_queries")
        result["sub_queries"] = (
            [str(q).strip() for q in sq if str(q).strip()] if isinstance(sq, list) else []
        )
        if not result["sub_queries"]:
            result["sub_queries"] = [question]

        # 合并原问题 + 子问题作为向量检索的扩展查询
        all_queries = [question] + [
            q for q in result["sub_queries"] if q != question
        ]
        result["expanded_query"] = "；".join(all_queries)

        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[Dict]:
        return self._last_result


# ========== 测试 ==========

if __name__ == "__main__":
    processor = QueryProcessor()
    tests = [
        "格林公式是什么",
        "总结一下矩阵对角化的方法",
        "偏导数和全微分有什么区别，怎么用",
    ]
    for q in tests:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        r = processor.process(q)
        print(f"意图: {r['intent']}")
        print(f"关键词: {r['keywords']}")
        print(f"子问题: {r['sub_queries']}")
        print(f"扩展查询: {r['expanded_query'][:120]}...")
