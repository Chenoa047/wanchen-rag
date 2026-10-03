from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from wanchen_rag.bm25 import read_chunks, searchable_text
from wanchen_rag.paths import resolve_chunks_path


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_PROMPT = (
    "Instruct: Given a Chinese listed-company annual-report question, "
    "retrieve passages that contain the facts needed to answer it.\nQuery: "
)


def choose_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(
    cache_dir: Path,
    device: str | None = None,
    local_files_only: bool = False,
) -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(cache_dir),
        device=device or choose_device(),
        local_files_only=local_files_only,
    )
    model.max_seq_length = 2048
    return model


def _chunk_batches(
    path: Path,
    batch_size: int,
    skip: int = 0,
) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for index, chunk in enumerate(read_chunks(path)):
        if index < skip:
            continue
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_vector_index(
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    batch_size: int = 4,
    device: str | None = None,
    local_files_only: bool = False,
) -> dict[str, object]:
    with chunks_path.open(encoding="utf-8") as stream:
        count = sum(1 for _ in stream)
    if count == 0:
        raise ValueError("没有可编码的检索块")

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    matrix: np.memmap | None = None
    offset = 0

    if embeddings_path.exists():
        if metadata_path.exists():
            raise FileExistsError("向量索引已完整存在，为避免误用旧向量已停止覆盖")
        existing = np.load(embeddings_path, mmap_mode="r+")
        if existing.shape != (count, 1024) or existing.dtype != np.float32:
            raise ValueError("已有向量文件的形状或类型与当前语料不匹配")
        filled = np.linalg.norm(existing, axis=1) > 0.5
        offset = int(filled.sum())
        if filled[:offset].all() and not filled[offset:].any():
            matrix = existing
            print(f"检测到可续跑的向量文件，从 {offset}/{count} 继续", flush=True)
        else:
            raise ValueError("已有向量文件不是连续写入，为避免损坏结果已停止续跑")

    model = load_model(
        cache_dir,
        device=device,
        local_files_only=local_files_only,
    )

    for batch in _chunk_batches(chunks_path, batch_size, skip=offset):
        texts = [searchable_text(chunk) for chunk in batch]
        encoded = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(count, encoded.shape[1]),
            )
        matrix[offset : offset + len(batch)] = encoded
        offset += len(batch)
        if offset == count or offset % (batch_size * 100) == 0:
            print(f"向量编码进度：{offset}/{count}", flush=True)

    if matrix is None:
        raise RuntimeError("向量矩阵未创建")
    matrix.flush()
    metadata = {
        "model_name": MODEL_NAME,
        "query_prompt": QUERY_PROMPT,
        "count": count,
        "dimension": int(matrix.shape[1]),
        "dtype": "float32",
        "chunks_path": os.path.relpath(chunks_path.resolve(), metadata_path.parent.resolve()),
        "device_used": str(model.device),
        "batch_size": batch_size,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class VectorIndex:
    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        cache_dir: Path,
        device: str | None = None,
    ) -> None:
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.chunks = list(read_chunks(resolve_chunks_path(self.metadata, metadata_path)))
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("向量数量与检索块数量不一致")
        self.model = load_model(cache_dir, device=device, local_files_only=True)

    def search(
        self,
        query: str,
        top_k: int = 8,
        stock_code: str | None = None,
    ) -> list[dict[str, object]]:
        query_vector = self.model.encode(
            [query],
            prompt=QUERY_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        scores = self.embeddings @ query_vector
        candidates = np.arange(len(scores))
        if stock_code is not None:
            candidates = np.array(
                [
                    i
                    for i, chunk in enumerate(self.chunks)
                    if chunk["stock_code"] == stock_code
                ],
                dtype=np.int64,
            )
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")[:top_k]]
        return [
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk": self.chunks[int(index)],
            }
            for rank, index in enumerate(ranked, start=1)
        ]
