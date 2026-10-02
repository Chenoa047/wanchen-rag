import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.embedding import _chunk_batches, build_vector_index


class EmbeddingTests(unittest.TestCase):
    def test_chunk_batches_can_skip_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"chunk_id": str(index)}) + "\n" for index in range(5)
                ),
                encoding="utf-8",
            )

            batches = list(_chunk_batches(path, batch_size=2, skip=3))

        self.assertEqual(
            [[row["chunk_id"] for row in batch] for batch in batches],
            [["3", "4"]],
        )

    def test_completed_index_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            embeddings = root / "embeddings.npy"
            metadata = root / "embeddings.json"
            chunks.write_text(json.dumps({"chunk_id": "1"}) + "\n", encoding="utf-8")
            np.save(embeddings, np.ones((1, 1024), dtype=np.float32))
            metadata.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                build_vector_index(
                    chunks_path=chunks,
                    embeddings_path=embeddings,
                    metadata_path=metadata,
                    cache_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
