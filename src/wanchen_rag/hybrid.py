from __future__ import annotations

from dataclasses import dataclass

from wanchen_rag.bm25 import Bm25Index
from wanchen_rag.embedding import VectorIndex


@dataclass(frozen=True)
class HybridResult:
    rank: int
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    vector_rank: int | None
    vector_score: float | None
    chunk: dict[str, object]


def rrf_fuse(
    bm25_rows: list[tuple[int, float, dict[str, object]]],
    vector_rows: list[tuple[int, float, dict[str, object]]],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[HybridResult]:
    combined: dict[str, dict[str, object]] = {}
    for source, rows in (("bm25", bm25_rows), ("vector", vector_rows)):
        for rank, score, chunk in rows:
            chunk_id = str(chunk["chunk_id"])
            entry = combined.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "vector_rank": None,
                    "vector_score": None,
                },
            )
            entry["rrf_score"] = float(entry["rrf_score"]) + 1.0 / (rrf_k + rank)
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = score

    ordered = sorted(
        combined.values(),
        key=lambda item: (-float(item["rrf_score"]), str(item["chunk"]["chunk_id"])),
    )[:top_k]
    return [
        HybridResult(
            rank=rank,
            rrf_score=float(item["rrf_score"]),
            bm25_rank=item["bm25_rank"],
            bm25_score=item["bm25_score"],
            vector_rank=item["vector_rank"],
            vector_score=item["vector_score"],
            chunk=item["chunk"],
        )
        for rank, item in enumerate(ordered, start=1)
    ]


class HybridIndex:
    def __init__(self, bm25_index: Bm25Index, vector_index: VectorIndex) -> None:
        self.bm25_index = bm25_index
        self.vector_index = vector_index

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 20,
        stock_code: str | None = None,
    ) -> list[HybridResult]:
        bm25_results = self.bm25_index.search(
            query,
            top_k=candidate_k,
            stock_code=stock_code,
        )
        vector_results = self.vector_index.search(
            query,
            top_k=candidate_k,
            stock_code=stock_code,
        )
        return rrf_fuse(
            bm25_rows=[(row.rank, row.score, row.chunk) for row in bm25_results],
            vector_rows=[
                (int(row["rank"]), float(row["score"]), row["chunk"])
                for row in vector_results
            ],
            top_k=top_k,
        )

    def stratified_search(
        self,
        query: str,
        stock_codes: list[str],
        per_company: int = 2,
        candidate_k: int = 20,
    ) -> list[HybridResult]:
        combined: list[HybridResult] = []
        for stock_code in stock_codes:
            company_results = self.search(
                query,
                top_k=per_company,
                candidate_k=candidate_k,
                stock_code=stock_code,
            )
            for result in company_results:
                combined.append(
                    HybridResult(
                        rank=len(combined) + 1,
                        rrf_score=result.rrf_score,
                        bm25_rank=result.bm25_rank,
                        bm25_score=result.bm25_score,
                        vector_rank=result.vector_rank,
                        vector_score=result.vector_score,
                        chunk=result.chunk,
                    )
                )
        return combined
