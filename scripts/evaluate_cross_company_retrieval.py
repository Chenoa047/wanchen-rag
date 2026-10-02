from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.bm25 import load_index  # noqa: E402
from wanchen_rag.embedding import VectorIndex  # noqa: E402
from wanchen_rag.hybrid import HybridIndex, HybridResult  # noqa: E402


QUESTIONS = (
    (
        "Q8",
        "十家公司2025年营业收入和归母净利润如何排名？",
    ),
    (
        "Q9",
        "十家公司2025年营业收入同比增速排名如何，哪家增长最快、哪家出现下滑？",
    ),
    (
        "Q10",
        "万辰集团、三只松鼠、良品铺子和来伊份的渠道模式及线下门店策略有何不同？",
    ),
)

Q10_STOCK_CODES = ["300972", "300783", "603719", "603777"]


def serialize(result: HybridResult) -> dict[str, object]:
    return {
        "rank": result.rank,
        "rrf_score": result.rrf_score,
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
        "company_name": result.chunk["company_name"],
        "stock_code": result.chunk["stock_code"],
        "pdf_page": result.chunk["pdf_page"],
        "chunk_id": result.chunk["chunk_id"],
        "text": result.chunk["text"],
    }


def main() -> int:
    with (PROJECT_ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        stock_codes = [row["stock_code"] for row in csv.DictReader(stream)]

    index_dir = PROJECT_ROOT / "data" / "index"
    hybrid = HybridIndex(
        bm25_index=load_index(
            index_dir / "all_bm25.pkl",
            PROJECT_ROOT / "config" / "jieba_userdict.txt",
        ),
        vector_index=VectorIndex(
            embeddings_path=index_dir / "all_embeddings.npy",
            metadata_path=index_dir / "all_embeddings.json",
            cache_dir=PROJECT_ROOT / "data" / "models",
        ),
    )

    records = []
    failed = False
    for case_id, query in QUESTIONS:
        target_codes = stock_codes if case_id in {"Q8", "Q9"} else Q10_STOCK_CODES
        baseline = hybrid.search(query, top_k=8, candidate_k=20)
        stratified = hybrid.stratified_search(
            query,
            stock_codes=target_codes,
            per_company=2,
            candidate_k=20,
        )
        baseline_companies = sorted({str(row.chunk["stock_code"]) for row in baseline})
        baseline_target_companies = sorted(set(baseline_companies).intersection(target_codes))
        stratified_companies = sorted({str(row.chunk["stock_code"]) for row in stratified})
        status = "PASS" if stratified_companies == sorted(target_codes) else "FAIL"
        print(
            f"{status} {case_id}: baseline 覆盖 {len(baseline_target_companies)}/{len(target_codes)} 家，"
            f"分层召回覆盖 {len(stratified_companies)}/{len(target_codes)} 家"
        )
        records.append(
            {
                "case_id": case_id,
                "query": query,
                "target_stock_codes": target_codes,
                "baseline": {
                    "company_coverage": baseline_companies,
                    "target_company_coverage": baseline_target_companies,
                    "results": [serialize(row) for row in baseline],
                },
                "stratified": {
                    "company_coverage": stratified_companies,
                    "results": [serialize(row) for row in stratified],
                },
            }
        )
        failed = failed or status == "FAIL"

    output_path = PROJECT_ROOT / "evaluation" / "cross_company_retrieval.json"
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测记录：{output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
