from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.bm25 import load_index  # noqa: E402
from wanchen_rag.embedding import VectorIndex  # noqa: E402
from wanchen_rag.hybrid import HybridIndex  # noqa: E402


CASES = (
    ("Q1", "万辰集团2025年营业收入和归母净利润分别是多少，同比增长多少？", {8}),
    ("Q2", "万辰集团2025年量贩零食与食用菌业务的收入和毛利率分别是多少？", {20, 21, 285}),
    ("Q3", "万辰集团2025年末门店数量是多少，报告期内新增与减少门店各多少家？", {15}),
    ("Q4", "万辰集团2025年销售毛利率和销售费用率比上年变化了多少？", {8, 21, 24}),
    ("Q5", "万辰集团2025年经营活动现金流量净额与合并净利润是否匹配？", {27, 292}),
    ("Q6", "万辰集团面临哪些门店、库存周转和食品安全风险，如何应对？", {32, 33}),
    ("Q7", "万辰集团2026年量贩零食业务的主要经营计划是什么？", {31}),
    (
        "Q11",
        "万辰集团审计报告称2025年营业收入为5,145,914.86万元，主要财务指标表称为51,459,148,553.51元，两者是否矛盾？",
        {8, 194},
    ),
)


def main() -> int:
    index_dir = PROJECT_ROOT / "data" / "index"
    hybrid = HybridIndex(
        bm25_index=load_index(
            index_dir / "wanchen_bm25.pkl",
            PROJECT_ROOT / "config" / "jieba_userdict.txt",
        ),
        vector_index=VectorIndex(
            embeddings_path=index_dir / "wanchen_embeddings.npy",
            metadata_path=index_dir / "wanchen_embeddings.json",
            cache_dir=PROJECT_ROOT / "data" / "models",
        ),
    )

    records = []
    failed = False
    for case_id, query, expected_pages in CASES:
        results = hybrid.search(query, top_k=8, candidate_k=20)
        pages = [int(result.chunk["pdf_page"]) for result in results]
        missing = expected_pages.difference(pages)
        status = "PASS" if not missing else "FAIL"
        print(f"{status} {case_id}: top-8 页码={pages}，目标页={sorted(expected_pages)}")
        records.append(
            {
                "case_id": case_id,
                "query": query,
                "expected_pages": sorted(expected_pages),
                "status": status,
                "results": [
                    {
                        "rrf_rank": result.rank,
                        "rrf_score": result.rrf_score,
                        "bm25_rank": result.bm25_rank,
                        "bm25_score": result.bm25_score,
                        "vector_rank": result.vector_rank,
                        "vector_score": result.vector_score,
                        "chunk_id": result.chunk["chunk_id"],
                        "pdf_page": result.chunk["pdf_page"],
                        "chapter": result.chunk["chapter"],
                        "content_type": result.chunk["content_type"],
                        "text": result.chunk["text"],
                    }
                    for result in results
                ],
            }
        )
        failed = failed or bool(missing)

    output_path = PROJECT_ROOT / "evaluation" / "wanchen_retrieval_local_questions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测记录：{output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
