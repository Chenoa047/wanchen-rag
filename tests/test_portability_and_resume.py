import json
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from wanchen_rag.evaluation import generation_succeeded, evaluation_matches
from wanchen_rag.paths import resolve_report_path, resolve_chunks_path


class PortabilityAndResumeTests(unittest.TestCase):
    def test_completed_batch_does_not_access_environment_or_model(self):
        from scripts import run_answer_evaluation as runner
        with patch.object(runner, "os", SimpleNamespace()), patch.object(runner, "HybridIndex", side_effect=AssertionError("不应加载模型")):
            self.assertEqual(runner.main(), 0)

    def test_old_grade_cannot_attach_to_regenerated_answer(self):
        answer = {"answer": "重新生成的回答", "generation_status": "success"}
        self.assertFalse(evaluation_matches(answer, {"answer_status": "正确"}))
        digest = hashlib.sha256(answer["answer"].encode()).hexdigest()
        self.assertTrue(evaluation_matches(answer, {"answer_sha256": digest}))
        self.assertFalse(evaluation_matches({**answer, "generation_status": "failed"}, {"answer_sha256": digest}))

    def test_partial_answers_are_successful_requests(self):
        for error in ("未召回", "跨公司覆盖不足", "生成误读"):
            self.assertTrue(generation_succeeded({"answer": "已有回答", "error_type": error}))

    def test_all_shipped_answers_should_be_skipped_on_resume(self):
        records = json.loads((PROJECT_ROOT / "evaluation/model_answers.json").read_text())
        self.assertEqual(len(records), 13)
        self.assertTrue(all(generation_succeeded(record) for record in records))

    def test_new_and_legacy_api_failures_are_retried(self):
        for record in (
            None, {}, {"answer": " "},
            {"answer": "失败提示", "error_type": "APITimeoutError"},
            {"answer": "失败提示", "generation_error": "APIConnectionError"},
            {"answer": "失败提示", "generation_status": "failed"},
            {"answer": "失败提示", "manual_evaluation": "待重新生成"},
        ):
            self.assertFalse(generation_succeeded(record))

    def test_portable_pdf_preferred_over_original_computer_path(self):
        row = {"file_name": "report.pdf", "local_path": "/old-machine/report.pdf"}
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(resolve_report_path(row, Path("/repo")), Path("/repo/data/raw/report.pdf"))
        with patch.object(Path, "is_file", return_value=False):
            self.assertEqual(resolve_report_path(row, Path("/repo")), Path("/old-machine/report.pdf"))
            self.assertEqual(resolve_report_path({"file_name": "report.pdf", "local_path": "reports/report.pdf"}, Path("/repo")), Path("/repo/reports/report.pdf"))

    def test_moved_vector_index_supports_relative_and_legacy_paths(self):
        metadata_path = Path("/new-repo/data/index/all_embeddings.json")
        relative = resolve_chunks_path({"chunks_path": "../processed/all_chunks.jsonl"}, metadata_path)
        self.assertEqual(relative.resolve(), Path("/new-repo/data/processed/all_chunks.jsonl"))
        with patch.object(Path, "is_file", return_value=False):
            old = resolve_chunks_path({"chunks_path": "/old-repo/data/processed/all_chunks.jsonl"}, metadata_path)
        self.assertEqual(old, Path("/new-repo/data/processed/all_chunks.jsonl"))


if __name__ == "__main__":
    unittest.main()
