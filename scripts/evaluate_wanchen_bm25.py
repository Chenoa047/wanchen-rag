from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.bm25 import load_index  # noqa: E402


CASES = (
    (
        "Q1",
        "万辰集团2025年营业收入和归母净利润分别是多少，同比增长多少？",
        {8},
    ),
    (
        "Q2",
        "万辰集团2025年量贩零食与食用菌业务的收入和毛利率分别是多少？",
        {20, 21, 285},
    ),
    (
        "Q3",
        "万辰集团2025年末门店数量是多少，报告期内新增与减少门店各多少家？",
        {15},
    ),
)


def main() -> int:
    index = load_index(
        PROJECT_ROOT / "data" / "index" / "wanchen_bm25.pkl",
        PROJECT_ROOT / "config" / "jieba_userdict.txt",
    )
    failed = False
    for case_id, query, expected_pages in CASES:
        results = index.search(query, top_k=8)
        pages = [int(result.chunk["pdf_page"]) for result in results]
        missing = expected_pages.difference(pages)
        status = "PASS" if not missing else "FAIL"
        print(f"{status} {case_id}: top-8 页码={pages}，目标页={sorted(expected_pages)}")
        failed = failed or bool(missing)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
