from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.extract import (
    chapter_for_page,
    detect_chapter,
    printed_page_from_text,
    split_text,
    table_to_lines,
)
from wanchen_rag.bm25 import expand_query, make_tokenizer, tokenize
from wanchen_rag.hybrid import rrf_fuse
from wanchen_rag.answering import citation


class ExtractTests(unittest.TestCase):
    def test_chapter_mapping_uses_pdf_pages(self) -> None:
        self.assertEqual(chapter_for_page(15), "第三节 管理层讨论与分析")
        self.assertEqual(chapter_for_page(193), "第八节 财务报告")

    def test_generic_chapter_detection_ignores_table_of_contents(self) -> None:
        current = "封面及报告信息"
        toc = "目录\n第一节 重要提示\n第二节 公司简介\n第三节 管理层讨论"
        self.assertEqual(detect_chapter(toc, current), current)
        self.assertEqual(
            detect_chapter("第三节 管理层讨论与分析\n一、主要业务", current),
            "第三节 管理层讨论与分析",
        )
        self.assertEqual(
            detect_chapter("第七节 债券相关情况\n不适用\n第八节 财务报告", current),
            "第八节 财务报告",
        )

    def test_printed_page_is_only_taken_from_last_line(self) -> None:
        self.assertEqual(printed_page_from_text("正文\n2025年\n15"), 15)
        self.assertEqual(printed_page_from_text("正文\n41/210"), 41)
        self.assertIsNone(printed_page_from_text("正文包含15但末行不是页码"))

    def test_table_rows_repeat_column_meaning(self) -> None:
        rows = [
            ["地区", "期初门店数量", "新增门店数量", "期末门店数量"],
            ["合计", "14,196", "4,720", "18,314"],
        ]
        self.assertEqual(
            table_to_lines(rows),
            ["地区=合计；期初门店数量=14,196；新增门店数量=4,720；期末门店数量=18,314"],
        )

    def test_two_level_table_header_is_combined(self) -> None:
        rows = [
            ["", "2025年", None, "2024年", None, "同比增减"],
            [None, "金额", "占比", "金额", "占比", None],
            ["量贩零食", "50", "98.83%", "31", "98.33%", "59.98%"],
        ]
        line = table_to_lines(rows)[0]
        self.assertIn("2025年 金额=50", line)
        self.assertIn("2025年 占比=98.83%", line)
        self.assertIn("同比增减=59.98%", line)

    def test_text_chunks_do_not_exceed_page_boundary(self) -> None:
        chunks = split_text("第一段。" * 300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertTrue(all(len(chunk) <= 800 for chunk in chunks))

    def test_financial_terms_are_preserved_by_tokenizer(self) -> None:
        tokenizer = make_tokenizer(PROJECT_ROOT / "config" / "jieba_userdict.txt")
        tokens = tokenize("万辰集团的营业收入和归母净利润", tokenizer)
        self.assertIn("万辰集团", tokens)
        self.assertIn("营业收入", tokens)
        self.assertIn("归母净利润", tokens)

    def test_query_synonym_expansion(self) -> None:
        expanded = expand_query("营业收入和归母净利润是多少")
        self.assertIn("归属于上市公司股东的净利润", expanded)

    def test_rrf_rewards_chunks_found_by_both_retrievers(self) -> None:
        shared = {"chunk_id": "shared"}
        only_bm25 = {"chunk_id": "bm25-only"}
        only_vector = {"chunk_id": "vector-only"}
        results = rrf_fuse(
            bm25_rows=[(1, 8.0, only_bm25), (2, 7.0, shared)],
            vector_rows=[(1, 0.9, only_vector), (2, 0.8, shared)],
            top_k=3,
        )
        self.assertEqual(results[0].chunk["chunk_id"], "shared")
        self.assertEqual(results[0].bm25_rank, 2)
        self.assertEqual(results[0].vector_rank, 2)

    def test_citation_contains_verifiable_source(self) -> None:
        chunk = {
            "company_name": "万辰集团",
            "report_year": 2025,
            "pdf_page": 15,
            "chunk_id": "300972-p015-table-02-01",
        }
        self.assertEqual(
            citation(chunk),
            "[万辰集团｜2025年报｜PDF第15页｜300972-p015-table-02-01]",
        )


if __name__ == "__main__":
    unittest.main()
