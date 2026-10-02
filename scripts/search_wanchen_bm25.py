from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.bm25 import load_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检索万辰集团 2025 年报")
    parser.add_argument("query", help="问题或检索词")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    index = load_index(
        PROJECT_ROOT / "data" / "index" / "wanchen_bm25.pkl",
        PROJECT_ROOT / "config" / "jieba_userdict.txt",
    )
    for result in index.search(args.query, top_k=args.top_k):
        chunk = result.chunk
        preview = str(chunk["text"]).replace("\n", " ")[:220]
        print(
            f"#{result.rank} score={result.score:.4f} "
            f"PDF第{chunk['pdf_page']}页 {chunk['chunk_id']}\n{preview}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

