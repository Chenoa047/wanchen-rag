from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wanchen_rag.paths import resolve_report_path  # noqa: E402

REQUIRED_METADATA = {
    "company_name",
    "stock_code",
    "report_year",
    "report_type",
    "chapter",
    "pdf_page",
    "printed_page",
    "source_pdf",
}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> int:
    with (PROJECT_ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))

    failed = False
    totals = {"pages": 0, "tables": 0, "chunks": 0}
    for row in manifest:
        output_dir = PROJECT_ROOT / "data" / "processed" / row["stock_code"]
        try:
            pages = load_jsonl(output_dir / "pages.jsonl")
            tables = load_jsonl(output_dir / "tables.jsonl")
            chunks = load_jsonl(output_dir / "chunks.jsonl")
            errors = []
            if len(pages) != int(row["page_count"]):
                errors.append(f"页数 {len(pages)} != {row['page_count']}")
            if not tables:
                errors.append("未提取到表格")
            if not chunks:
                errors.append("未生成检索块")
            if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
                errors.append("chunk_id 重复")
            for chunk in chunks:
                if REQUIRED_METADATA.difference(chunk):
                    errors.append("存在缺少元数据的块")
                    break
                if chunk["stock_code"] != row["stock_code"]:
                    errors.append("股票代码串库")
                    break
                if Path(str(chunk["source_pdf"])).resolve() != resolve_report_path(row, PROJECT_ROOT).resolve():
                    errors.append("来源文件串库")
                    break
        except FileNotFoundError as error:
            pages, tables, chunks = [], [], []
            errors = [f"缺少输出文件：{error.filename}"]

        totals["pages"] += len(pages)
        totals["tables"] += len(tables)
        totals["chunks"] += len(chunks)
        status = "PASS" if not errors else "FAIL"
        print(
            f"{status} {row['stock_code']} {row['company_name']}："
            f"{len(pages)} 页，{len(tables)} 表，{len(chunks)} 块"
            + (f"；{'；'.join(errors)}" if errors else "")
        )
        failed = failed or bool(errors)

    print(
        f"TOTAL：{totals['pages']} 页，{totals['tables']} 表，"
        f"{totals['chunks']} 个检索块"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
