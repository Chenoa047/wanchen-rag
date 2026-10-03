from pathlib import Path
import json
import sys
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def isolated_app(*, missing: bool = False) -> AppTest:
    # 只用替身环境，不读取运行测试的终端中的任何密钥。
    return AppTest.from_string(f'''
import app
from types import SimpleNamespace
from unittest.mock import Mock, patch
import streamlit as st
chunk = {{"company_name": "万辰集团", "stock_code": "300972", "report_year": 2025,
         "chapter": "主要财务指标", "pdf_page": 8, "chunk_id": "test-chunk",
         "content_type": "table", "text": "测试证据"}}
result = SimpleNamespace(rank=1, rrf_score=0.03, bm25_rank=1, vector_rank=1, chunk=chunk)
hybrid = Mock()
def search(*args, **kwargs):
    st.session_state["search_calls"] = st.session_state.get("search_calls", 0) + 1
    return [result]
hybrid.search.side_effect = search
hybrid.stratified_search.side_effect = search
with patch.object(app, "os", SimpleNamespace(environ={{}})), \\
     patch.object(app, "generate_answer", side_effect=AssertionError("不允许真实调用")), \\
     patch.object(app, "load_hybrid_index", return_value=hybrid), \\
     patch.object(app, "missing_local_files", return_value={['data/index/all_bm25.pkl'] if missing else []}):
    app.main()
''')


class AppSmokeTests(unittest.TestCase):
    def test_initial_page_loads_without_answer_service(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "2025 年报知识问答库")
        self.assertEqual(app.selectbox(key="scope").value, "全行业")
        self.assertEqual(app.metric[0].value, "10")
        self.assertEqual(app.metric[1].value, "2,044")
        expected_chunks = "6,800" if (PROJECT_ROOT / "data/processed/300972/chunks.jsonl").is_file() else "未完整建库"
        self.assertEqual(app.metric[2].value, expected_chunks)

    def test_all_saved_routes_can_be_viewed_without_model_or_api(self) -> None:
        app = isolated_app().run(timeout=30)
        records = json.loads((PROJECT_ROOT / "evaluation/model_answers.json").read_text())
        for index, record in enumerate(records):
            app.selectbox(key="saved_record").select(index).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.warning)
            self.assertTrue(any(record["answer"] == item.value for item in app.markdown))
        self.assertNotIn("search_calls", app.session_state)

    def test_missing_index_shows_instructions(self) -> None:
        app = isolated_app(missing=True).run(timeout=30)
        app.radio[0].set_value("本地检索与提问").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("尚未建好" in item.value for item in app.warning))

    def test_live_results_survive_rerun_and_clear_on_scope_change(self) -> None:
        app = isolated_app().run(timeout=30)
        app.radio[0].set_value("本地检索与提问").run()
        app.text_area[0].set_value("万辰集团营业收入是多少？").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["search_calls"], 1)
        app.run()
        self.assertEqual(app.session_state["search_calls"], 1)
        self.assertTrue(any(item.value == "测试证据" for item in app.markdown))
        app.selectbox(key="scope").select("三只松鼠").run()
        self.assertNotIn("search_result", app.session_state)
        self.assertFalse(any(item.value == "测试证据" for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
