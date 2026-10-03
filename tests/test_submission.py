import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def test_all_answer_evidence_and_citations_are_published(self):
        answers = json.loads((ROOT / "evaluation/model_answers.json").read_text(encoding="utf-8"))
        chunks = json.loads((ROOT / "evaluation/answer_evidence.json").read_text(encoding="utf-8"))["chunks"]
        for answer in answers:
            self.assertTrue(set(answer["chunk_ids"]).issubset(chunks))
            for company, year, page, chunk_id in re.findall(
                r"\[([^｜\[\]]+)｜(\d+)年报｜PDF第(\d+)页｜([^\]]+)\]", answer["answer"]
            ):
                self.assertIn(chunk_id, answer["chunk_ids"])
                chunk = chunks[chunk_id]
                self.assertEqual((company, int(year), int(page)),
                                 (chunk["company_name"], chunk["report_year"], chunk["pdf_page"]))

    def test_readable_review_preserves_answers_and_links(self):
        review = (ROOT / "evaluation/REVIEW.md").read_text(encoding="utf-8")
        evidence = (ROOT / "evaluation/EVIDENCE.md").read_text(encoding="utf-8")
        answers = json.loads((ROOT / "evaluation/model_answers.json").read_text(encoding="utf-8"))
        for answer in answers:
            self.assertIn(answer["answer"], review)
            for chunk_id in answer["chunk_ids"]:
                self.assertIn(f"## {chunk_id}\n", evidence)
        for filename in ("README.md", "evaluation/REVIEW.md", "evaluation/EVIDENCE.md"):
            path = ROOT / filename
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if not link.startswith(("https://", "http://", "#")):
                    self.assertTrue((path.parent / link.split("#")[0]).is_file(), link)


if __name__ == "__main__":
    unittest.main()
