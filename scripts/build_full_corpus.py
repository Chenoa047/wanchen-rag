from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with (PROJECT_ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    output_path = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
    seen: set[str] = set()
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            chunk_path = (
                PROJECT_ROOT / "data" / "processed" / row["stock_code"] / "chunks.jsonl"
            )
            with chunk_path.open(encoding="utf-8") as stream:
                for line in stream:
                    chunk = json.loads(line)
                    chunk_id = str(chunk["chunk_id"])
                    if chunk_id in seen:
                        raise ValueError(f"重复 chunk_id：{chunk_id}")
                    seen.add(chunk_id)
                    output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count += 1
    print(f"全库语料合并完成：{len(rows)} 家公司，{count} 个检索块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

