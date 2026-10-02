from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pdfplumber


RUNNING_HEADER = re.compile(r"^.*2025\s*年年度报告全文$")
PAGE_NUMBER = re.compile(r"^\d+$")
PAGE_FRACTION = re.compile(r"^(\d+)/\d+$")
UNIT = re.compile(r"单位[：:]\s*[^\s]+")
CHAPTER_HEADING = re.compile(r"^(第[一二三四五六七八九十]+节\s+[^\n]+)")

WANCHEN_CHAPTERS = (
    (2, "第一节 重要提示、目录和释义"),
    (7, "第二节 公司简介和主要财务指标"),
    (11, "第三节 管理层讨论与分析"),
    (35, "第四节 公司治理、环境和社会"),
    (59, "第五节 重要事项"),
    (183, "第六节 股份变动及股东情况"),
    (192, "第七节 债券相关情况"),
    (193, "第八节 财务报告"),
)


@dataclass(frozen=True)
class ReportMetadata:
    stock_code: str
    company_name: str
    report_year: int
    report_type: str
    source_pdf: str


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def clean_page_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or RUNNING_HEADER.fullmatch(line) or PAGE_NUMBER.fullmatch(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def printed_page_from_text(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if PAGE_NUMBER.fullmatch(lines[-1]):
        value = int(lines[-1])
    else:
        match = PAGE_FRACTION.fullmatch(lines[-1])
        if not match:
            return None
        value = int(match.group(1))
    return value if 0 < value < 1000 else None


def detect_chapter(text: str, current: str) -> str:
    if any(line.strip() == "目录" for line in text.splitlines()[:15]):
        return current
    headings = []
    for line in text.splitlines():
        match = CHAPTER_HEADING.match(line.strip())
        if match:
            headings.append(re.sub(r"\s+", " ", match.group(1)).strip())
    unique = list(dict.fromkeys(headings))
    return unique[-1] if unique else current


def chapter_for_page(pdf_page: int) -> str:
    chapter = "封面及报告信息"
    for start_page, name in WANCHEN_CHAPTERS:
        if pdf_page < start_page:
            break
        chapter = name
    return chapter


def _fill_merged_headers(values: Sequence[str]) -> list[str]:
    filled: list[str] = []
    current = ""
    for value in values:
        if value:
            current = value
        filled.append(current)
    return filled


def table_to_lines(rows: Sequence[Sequence[object]]) -> list[str]:
    """Convert a PDF table to searchable rows with headers repeated inline."""
    if not rows:
        return []

    grid = [[clean_cell(cell) for cell in row] for row in rows]
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    first = grid[0]
    use_two_headers = len(grid) > 1 and any(not cell for cell in first[1:])
    if use_two_headers:
        first_filled = _fill_merged_headers(first)
        second = grid[1]
        headers = [
            " ".join(part for part in (first_filled[i], second[i]) if part).strip()
            for i in range(width)
        ]
        data_rows = grid[2:]
    else:
        headers = first
        data_rows = grid[1:]

    headers = [header or ("项目" if i == 0 else f"列{i + 1}") for i, header in enumerate(headers)]
    group = ""
    output: list[str] = []
    for row in data_rows:
        populated = [(i, value) for i, value in enumerate(row) if value]
        if not populated:
            continue
        if len(populated) == 1 and width > 1:
            group = populated[0][1]
            continue
        fields = []
        if group:
            fields.append(f"分类={group}")
        fields.extend(
            f"{headers[i]}={value}" for i, value in populated
        )
        output.append("；".join(fields))
    return output


def _table_context(page: pdfplumber.page.Page, table_bbox: tuple[float, float, float, float]) -> tuple[str, str]:
    top = table_bbox[1]
    context = page.crop((0, max(0, top - 110), page.width, top)).extract_text() or ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in context.splitlines() if line.strip()]
    unit = ""
    for line in reversed(lines):
        match = UNIT.search(line)
        if match:
            unit = match.group(0)
            break
    candidates = [line for line in lines if "单位" not in line and not RUNNING_HEADER.fullmatch(line)]
    title = candidates[-1] if candidates else "未识别表名"
    return title, unit


def split_text(text: str, target: int = 800, overlap: int = 120) -> list[str]:
    if len(text) <= target:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + target, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [text.rfind(mark, start + target // 2, hard_end) for mark in ("\n", "。", "；")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def group_table_lines(lines: Sequence[str], prefix: str, target: int = 1200) -> list[str]:
    groups: list[str] = []
    current = prefix
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if current != prefix and len(candidate) > target:
            groups.append(current)
            current = f"{prefix}\n{line}" if prefix else line
        else:
            current = candidate
    if current and current != prefix:
        groups.append(current)
    return groups


def load_metadata(manifest_path: Path, stock_code: str) -> ReportMetadata:
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["stock_code"] == stock_code:
                return ReportMetadata(
                    stock_code=stock_code,
                    company_name=row["company_name"],
                    report_year=int(row["report_year"]),
                    report_type=row["report_type"],
                    source_pdf=row["local_path"],
                )
    raise ValueError(f"清单中没有股票代码 {stock_code}")


def _base_record(
    metadata: ReportMetadata,
    pdf_page: int,
    chapter: str,
    printed_page: int | None,
) -> dict[str, object]:
    return {
        "company_name": metadata.company_name,
        "stock_code": metadata.stock_code,
        "report_year": metadata.report_year,
        "report_type": metadata.report_type,
        "chapter": chapter,
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "source_pdf": metadata.source_pdf,
    }


def extract_report(metadata: ReportMetadata, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_path = output_dir / "pages.jsonl"
    table_path = output_dir / "tables.jsonl"
    chunk_path = output_dir / "chunks.jsonl"
    counters = {"pages": 0, "tables": 0, "text_chunks": 0, "table_chunks": 0}

    with (
        pdfplumber.open(metadata.source_pdf) as pdf,
        page_path.open("w", encoding="utf-8") as pages_out,
        table_path.open("w", encoding="utf-8") as tables_out,
        chunk_path.open("w", encoding="utf-8") as chunks_out,
    ):
        current_chapter = "封面及报告信息"
        for pdf_page, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            text = clean_page_text(raw_text)
            if metadata.stock_code == "300972":
                current_chapter = chapter_for_page(pdf_page)
            else:
                current_chapter = detect_chapter(text, current_chapter)
            base = _base_record(
                metadata,
                pdf_page,
                current_chapter,
                printed_page_from_text(raw_text),
            )
            page_record = {**base, "content_type": "page", "text": text}
            pages_out.write(json.dumps(page_record, ensure_ascii=False) + "\n")
            counters["pages"] += 1

            for index, chunk in enumerate(split_text(text), start=1):
                chunk_record = {
                    **base,
                    "chunk_id": f"{metadata.stock_code}-p{pdf_page:03d}-text-{index:02d}",
                    "content_type": "text",
                    "table_title": "",
                    "text": chunk,
                }
                chunks_out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                counters["text_chunks"] += 1

            for table_index, table in enumerate(page.find_tables(), start=1):
                rows = table.extract() or []
                lines = table_to_lines(rows)
                if not lines:
                    continue
                title, unit = _table_context(page, table.bbox)
                table_record = {
                    **base,
                    "content_type": "table",
                    "table_index": table_index,
                    "table_title": title,
                    "unit": unit,
                    "rows": [[clean_cell(cell) for cell in row] for row in rows],
                    "searchable_rows": lines,
                }
                tables_out.write(json.dumps(table_record, ensure_ascii=False) + "\n")
                counters["tables"] += 1

                prefix = f"表名={title}" + (f"；{unit}" if unit else "")
                for group_index, table_text in enumerate(group_table_lines(lines, prefix), start=1):
                    chunk_record = {
                        **base,
                        "chunk_id": f"{metadata.stock_code}-p{pdf_page:03d}-table-{table_index:02d}-{group_index:02d}",
                        "content_type": "table",
                        "table_title": title,
                        "text": table_text,
                    }
                    chunks_out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                    counters["table_chunks"] += 1

    return counters


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)
