from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.extract import extract_report, load_metadata  # noqa: E402


def main() -> int:
    metadata = load_metadata(PROJECT_ROOT / "data_manifest.csv", "300972")
    output_dir = PROJECT_ROOT / "data" / "processed" / metadata.stock_code
    counters = extract_report(metadata, output_dir)
    print(f"万辰集团提取完成：{counters}")
    print(f"输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

