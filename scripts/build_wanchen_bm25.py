from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.bm25 import build_index  # noqa: E402


def main() -> int:
    count = build_index(
        chunks_path=PROJECT_ROOT / "data" / "processed" / "300972" / "chunks.jsonl",
        index_path=PROJECT_ROOT / "data" / "index" / "wanchen_bm25.pkl",
        user_dict=PROJECT_ROOT / "config" / "jieba_userdict.txt",
    )
    print(f"BM25 索引构建完成：{count} 个万辰集团检索块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

