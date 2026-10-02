from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.embedding import build_vector_index  # noqa: E402


def main() -> int:
    index_dir = PROJECT_ROOT / "data" / "index"
    metadata = build_vector_index(
        chunks_path=PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl",
        embeddings_path=index_dir / "all_embeddings.npy",
        metadata_path=index_dir / "all_embeddings.json",
        cache_dir=PROJECT_ROOT / "data" / "models",
        batch_size=4,
        local_files_only=True,
    )
    print(f"全库向量索引构建完成：{metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

