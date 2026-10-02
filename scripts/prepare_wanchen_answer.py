from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.answering import build_context  # noqa: E402
from wanchen_rag.bm25 import load_index  # noqa: E402
from wanchen_rag.embedding import VectorIndex  # noqa: E402
from wanchen_rag.hybrid import HybridIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成万辰财报问题的离线证据包")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

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
    results = hybrid.search(args.question, top_k=args.top_k, candidate_k=20)
    print(f"问题：{args.question}\n")
    print(build_context(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

