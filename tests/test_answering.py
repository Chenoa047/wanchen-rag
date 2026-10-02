from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.answering import build_messages, generate_answer  # noqa: E402
from wanchen_rag.hybrid import HybridResult  # noqa: E402


def sample_result() -> HybridResult:
    return HybridResult(
        rank=1,
        rrf_score=0.03,
        bm25_rank=1,
        bm25_score=8.0,
        vector_rank=2,
        vector_score=0.8,
        chunk={
            "company_name": "万辰集团",
            "stock_code": "300972",
            "report_year": 2025,
            "chapter": "第二节 公司简介和主要财务指标",
            "pdf_page": 8,
            "chunk_id": "300972-p008-table-03-01",
            "content_type": "table",
            "text": "营业收入=51,459,148,553.51元",
        },
    )


class AnsweringTests(unittest.TestCase):
    def test_context_contains_verifiable_citation(self) -> None:
        messages = build_messages("营业收入是多少？", [sample_result()])
        self.assertIn(
            "[万辰集团｜2025年报｜PDF第8页｜300972-p008-table-03-01]",
            messages[1]["content"],
        )

    @patch("openai.OpenAI")
    def test_generate_answer_uses_configured_model(self, openai_client: object) -> None:
        client = openai_client.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="答案 [引用]"))]
        )

        answer = generate_answer(
            "营业收入是多少？",
            [sample_result()],
            api_key="non-secret-placeholder",
            model="deepseek-flash",
        )

        self.assertEqual(answer, "答案 [引用]")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "deepseek-flash",
        )
        self.assertEqual(openai_client.call_args.kwargs["timeout"], 120.0)
        self.assertEqual(openai_client.call_args.kwargs["max_retries"], 0)


if __name__ == "__main__":
    unittest.main()
