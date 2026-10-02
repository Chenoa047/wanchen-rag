from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data_manifest.csv"
COMPANY_ALIASES = {
    "300972": ("万辰集团", "福建万辰食品集团股份有限公司"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return "".join(text.split())


def validate_row(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    pdf_path = Path(row["local_path"])

    if not pdf_path.is_file():
        return ["文件不存在"]

    if pdf_path.name != row["file_name"]:
        errors.append("文件名与清单不一致")

    if sha256(pdf_path) != row["sha256"]:
        errors.append("SHA-256 与清单不一致")

    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        errors.append("PDF 已加密")
        return errors

    if len(reader.pages) != int(row["page_count"]):
        errors.append("页数与清单不一致")

    first_pages = "\n".join(
        page.extract_text() or "" for page in reader.pages[:3]
    )
    compact = normalized(first_pages)

    company_names = COMPANY_ALIASES.get(
        row["stock_code"], (row["company_name"],)
    )
    if not any(name in compact for name in company_names):
        errors.append("首页未识别出公司简称")
    if "2025年年度报告" not in compact:
        errors.append("首页未识别出2025年年度报告")
    if "年度报告摘要" in compact:
        errors.append("疑似下载了年度报告摘要")

    return errors


def main() -> int:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 10:
        print(f"FAIL: 清单应有 10 行，实际为 {len(rows)} 行")
        return 1

    failed = False
    total_pages = 0
    for row in rows:
        errors = validate_row(row)
        total_pages += int(row["page_count"])
        if errors:
            failed = True
            print(f"FAIL {row['stock_code']} {row['company_name']}: {'；'.join(errors)}")
        else:
            print(
                f"PASS {row['stock_code']} {row['company_name']}: "
                f"{row['page_count']} 页"
            )

    print(f"TOTAL: {len(rows)} 份，{total_pages} 页")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
