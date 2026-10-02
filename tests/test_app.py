from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def test_initial_page_loads_without_answer_service(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "2025 年报知识问答库")
        self.assertEqual(app.selectbox[0].value, "全行业")
        self.assertEqual(app.metric[0].value, "10")
        self.assertEqual(app.metric[1].value, "2,044")
        self.assertEqual(app.metric[2].value, "6,800")


if __name__ == "__main__":
    unittest.main()
