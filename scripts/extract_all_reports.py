from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.extract import extract_report, load_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="顺序提取年报")
    parser.add_argument("--stock-code", action="append", help="只处理指定股票代码；可重复传入")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / "data_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        stock_codes = [row["stock_code"] for row in csv.DictReader(stream)]
    if args.stock_code:
        requested = set(args.stock_code)
        stock_codes = [code for code in stock_codes if code in requested]
        missing = requested.difference(stock_codes)
        if missing:
            parser.error(f"清单中没有股票代码：{', '.join(sorted(missing))}")

    for position, stock_code in enumerate(stock_codes, start=1):
        metadata = load_metadata(manifest_path, stock_code)
        counters = extract_report(
            metadata,
            PROJECT_ROOT / "data" / "processed" / stock_code,
        )
        print(
            f"[{position}/{len(stock_codes)}] {metadata.company_name}："
            f"{counters['pages']} 页，{counters['tables']} 表，"
            f"{counters['text_chunks'] + counters['table_chunks']} 块"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
