from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jieba
from rank_bm25 import BM25Okapi


TOKEN_CONTENT = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
QUERY_SYNONYMS = {
    "归母净利润": "归属于上市公司股东的净利润",
    "收入": "营业收入",
    "年末门店": "期末门店数量",
    "减少门店": "门店减少数量",
}


def make_tokenizer(user_dict: Path | None = None) -> jieba.Tokenizer:
    tokenizer = jieba.Tokenizer()
    if user_dict is not None:
        with user_dict.open(encoding="utf-8") as stream:
            tokenizer.load_userdict(stream)
    return tokenizer


def tokenize(text: str, tokenizer: jieba.Tokenizer) -> list[str]:
    return [
        token.lower().strip()
        for token in tokenizer.cut_for_search(text)
        if token.strip() and TOKEN_CONTENT.search(token)
    ]


def searchable_text(chunk: dict[str, object]) -> str:
    tags = (
        str(chunk["company_name"]),
        str(chunk["report_year"]),
        str(chunk["chapter"]),
        str(chunk.get("table_title", "")),
    )
    return " ".join((*tags, str(chunk["text"])))


def expand_query(query: str) -> str:
    expansions = [replacement for term, replacement in QUERY_SYNONYMS.items() if term in query]
    return " ".join((query, *expansions))


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk: dict[str, object]


class Bm25Index:
    def __init__(
        self,
        chunks: list[dict[str, object]],
        tokenized_corpus: list[list[str]],
        tokenizer: jieba.Tokenizer,
    ) -> None:
        self.chunks = chunks
        self.tokenized_corpus = tokenized_corpus
        self.tokenizer = tokenizer
        self.model = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 8,
        stock_code: str | None = None,
    ) -> list[SearchResult]:
        query_tokens = tokenize(expand_query(query), self.tokenizer)
        scores = self.model.get_scores(query_tokens)
        candidates = (
            range(len(scores))
            if stock_code is None
            else (
                i
                for i, chunk in enumerate(self.chunks)
                if chunk["stock_code"] == stock_code
            )
        )
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))[:top_k]
        return [
            SearchResult(rank=rank, score=float(scores[index]), chunk=self.chunks[index])
            for rank, index in enumerate(ranked, start=1)
        ]


def read_chunks(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def build_index(chunks_path: Path, index_path: Path, user_dict: Path) -> int:
    chunks = list(read_chunks(chunks_path))
    tokenizer = make_tokenizer(user_dict)
    tokenized_corpus = [tokenize(searchable_text(chunk), tokenizer) for chunk in chunks]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as stream:
        pickle.dump(
            {
                "chunks": chunks,
                "tokenized_corpus": tokenized_corpus,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return len(chunks)


def load_index(index_path: Path, user_dict: Path) -> Bm25Index:
    with index_path.open("rb") as stream:
        payload = pickle.load(stream)
    return Bm25Index(
        chunks=payload["chunks"],
        tokenized_corpus=payload["tokenized_corpus"],
        tokenizer=make_tokenizer(user_dict),
    )
