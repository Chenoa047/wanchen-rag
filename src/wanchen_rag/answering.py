from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wanchen_rag.hybrid import HybridResult


SYSTEM_PROMPT = """你是上市公司财报问答助手。只能依据提供的检索材料回答。
要求：
1. 每个事实结论后都附上材料中的完整引用标记；
2. 涉及计算时写出公式，并保证收入、成本等数据使用同一口径；
3. 若材料不足，明确写“现有召回材料未覆盖”，不要用常识补全；
4. 金额保留原始单位和精度，百分比与百分点不得混淆。
"""

REQUEST_TIMEOUT_SECONDS = 120.0


def citation(chunk: dict[str, object]) -> str:
    return (
        f"[{chunk['company_name']}｜{chunk['report_year']}年报｜"
        f"PDF第{chunk['pdf_page']}页｜{chunk['chunk_id']}]"
    )


def build_context(results: Sequence[HybridResult]) -> str:
    sections = []
    for result in results:
        chunk = result.chunk
        sections.append(
            "\n".join(
                (
                    f"材料{result.rank} {citation(chunk)}",
                    f"章节：{chunk['chapter']}｜类型：{chunk['content_type']}",
                    str(chunk["text"]),
                )
            )
        )
    return "\n\n".join(sections)


def build_messages(question: str, results: Sequence[HybridResult]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"问题：{question}\n\n检索材料：\n{build_context(results)}",
        },
    ]


def generate_answer(
    question: str,
    results: Sequence[HybridResult],
    *,
    api_key: str,
    model: str = "deepseek-flash",
) -> str:
    """Generate a grounded answer after the user has supplied their own key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, results),
        temperature=0.1,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "现有召回材料未覆盖"
