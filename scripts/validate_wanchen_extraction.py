from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "300972"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require_fact(chunks: list[dict[str, object]], pdf_page: int, facts: tuple[str, ...]) -> None:
    page_text = "\n".join(
        str(chunk["text"])
        for chunk in chunks
        if chunk["pdf_page"] == pdf_page and chunk["content_type"] == "table"
    )
    missing = [fact for fact in facts if fact not in page_text]
    if missing:
        raise AssertionError(f"PDF 第 {pdf_page} 页缺少关键字段：{missing}")


def main() -> int:
    pages = load_jsonl(OUTPUT_DIR / "pages.jsonl")
    tables = load_jsonl(OUTPUT_DIR / "tables.jsonl")
    chunks = load_jsonl(OUTPUT_DIR / "chunks.jsonl")

    if len(pages) != 338:
        raise AssertionError(f"应提取 338 页，实际 {len(pages)} 页")
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise AssertionError("chunk_id 存在重复")

    required_metadata = {
        "company_name",
        "stock_code",
        "report_year",
        "report_type",
        "chapter",
        "pdf_page",
        "printed_page",
        "source_pdf",
    }
    for chunk in chunks:
        missing = required_metadata.difference(chunk)
        if missing:
            raise AssertionError(f"{chunk.get('chunk_id', '未知块')} 缺少元数据：{sorted(missing)}")

    require_fact(
        chunks,
        8,
        (
            "营业收入（元）",
            "2025年=51,459,148,553.51",
            "本年比上年增减=59.17%",
            "归属于上市公司股东的净利润（元）",
            "2025年=1,344,598,814.55",
            "本年比上年增减=358.09%",
        ),
    )
    require_fact(
        chunks,
        15,
        (
            "期初门店数量=14,196",
            "新增门店数量=4,720",
            "门店减少数量（包括门店经营原因及非门店经营原因）=602",
            "期末门店数量=18,314",
        ),
    )
    require_fact(
        chunks,
        21,
        (
            "项目=量贩零食",
            "毛利率=12.32%",
            "毛利率比上年同期增减=1.46%",
        ),
    )

    print(
        "PASS 万辰集团提取验收："
        f"{len(pages)} 页，{len(tables)} 个表格，{len(chunks)} 个检索块；"
        "关键财务、门店与毛利率字段均可定位。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

