from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.answering import citation, generate_answer  # noqa: E402
from wanchen_rag.bm25 import load_index  # noqa: E402
from wanchen_rag.embedding import VectorIndex  # noqa: E402
from wanchen_rag.hybrid import HybridIndex  # noqa: E402


@st.cache_data
def load_manifest() -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@st.cache_data
def load_corpus_stats() -> list[dict[str, object]]:
    rows = []
    for report in load_manifest():
        chunks_path = (
            PROJECT_ROOT / "data" / "processed" / report["stock_code"] / "chunks.jsonl"
        )
        with chunks_path.open(encoding="utf-8") as stream:
            chunk_count = sum(1 for _ in stream)
        rows.append(
            {
                "公司": report["company_name"],
                "股票代码": report["stock_code"],
                "PDF页数": int(report["page_count"]),
                "检索块": chunk_count,
            }
        )
    return rows


@st.cache_data
def load_saved_answers() -> list[dict[str, object]]:
    answers_path = PROJECT_ROOT / "evaluation" / "model_answers.json"
    if not answers_path.exists():
        return []
    return json.loads(answers_path.read_text(encoding="utf-8"))


def find_saved_answer(question: str, route: str) -> str | None:
    normalized_question = "".join(question.split())
    for row in load_saved_answers():
        if row.get("route") != route:
            continue
        if "".join(str(row.get("question", "")).split()) == normalized_question:
            return str(row.get("answer", "")) or None
    return None


@st.cache_resource
def load_hybrid_index() -> HybridIndex:
    index_dir = PROJECT_ROOT / "data" / "index"
    return HybridIndex(
        bm25_index=load_index(
            index_dir / "all_bm25.pkl",
            PROJECT_ROOT / "config" / "jieba_userdict.txt",
        ),
        vector_index=VectorIndex(
            embeddings_path=index_dir / "all_embeddings.npy",
            metadata_path=index_dir / "all_embeddings.json",
            cache_dir=PROJECT_ROOT / "data" / "models",
        ),
    )


def main() -> None:
    st.set_page_config(page_title="2025 年报知识问答库", layout="wide")
    st.title("2025 年报知识问答库")
    st.caption("万辰集团为核心｜10 家休闲食品相关上市公司｜向量 + BM25 + RRF")

    manifest = load_manifest()
    corpus_stats = load_corpus_stats()
    company_to_code = {row["company_name"]: row["stock_code"] for row in manifest}
    total_pages = sum(int(row["page_count"]) for row in manifest)
    total_chunks = sum(int(row["检索块"]) for row in corpus_stats)

    with st.sidebar:
        st.header("检索设置")
        scope = st.selectbox("公司范围", ["全行业", *company_to_code])
        stratified = st.checkbox(
            "跨公司分层召回",
            value=scope == "全行业",
            disabled=scope != "全行业",
            help="每家公司至少召回 2 个证据块，防止单一公司垄断结果。",
        )
        st.metric("公司", len(manifest))
        st.metric("年报页数", f"{total_pages:,}")
        st.metric("检索块", f"{total_chunks:,}")

    with st.expander("语料统计（按公司）"):
        st.dataframe(pd.DataFrame(corpus_stats), width="stretch", hide_index=True)

    question = st.text_area(
        "请输入财报问题",
        placeholder="例如：十家公司 2025 年营业收入同比增速如何排名？",
        height=100,
    )
    submitted = st.button("检索证据", type="primary", disabled=not question.strip())
    if not submitted:
        st.info("无密钥时仍可检索证据；如果输入已完成评测的题目，页面会同时展示已保存的模型回答。")
        return

    with st.spinner("正在加载本地模型并检索……"):
        hybrid = load_hybrid_index()
        if scope == "全行业" and stratified:
            results = hybrid.stratified_search(
                question,
                stock_codes=[row["stock_code"] for row in manifest],
                per_company=2,
                candidate_k=20,
            )
        else:
            stock_code = company_to_code.get(scope)
            results = hybrid.search(
                question,
                top_k=8,
                candidate_k=20,
                stock_code=stock_code,
            )

    st.subheader("答案")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        try:
            with st.spinner("正在基于召回材料生成带引用的答案……"):
                answer = generate_answer(question, results, api_key=api_key, model=model)
            st.markdown(answer)
        except Exception:
            st.error("答案生成失败。请在终端检查脱敏后的报错信息；下方本地检索结果仍可使用。")
    else:
        route = (
            "分层召回，每公司2块"
            if scope == "全行业" and stratified
            else "RRF top-8 基线"
            if scope == "全行业"
            else "单公司 RRF top-8"
        )
        saved_answer = find_saved_answer(question, route)
        if saved_answer:
            st.caption("以下为本次批量评测已保存的模型回答（无密钥回放）。")
            st.markdown(saved_answer)
        else:
            st.info(
                "当前为无密钥检索模式，且该问题没有已保存的评测回答。"
                "您可自行配置 DEEPSEEK_API_KEY 后重启页面，以生成新回答。"
            )

    st.subheader("检索过程")
    ranking_rows = [
        {
            "RRF排名": result.rank,
            "公司": result.chunk["company_name"],
            "章节": result.chunk["chapter"],
            "PDF页": result.chunk["pdf_page"],
            "内容类型": result.chunk["content_type"],
            "BM25排名": result.bm25_rank,
            "向量排名": result.vector_rank,
            "RRF得分": round(result.rrf_score, 6),
            "块ID": result.chunk["chunk_id"],
        }
        for result in results
    ]
    st.dataframe(pd.DataFrame(ranking_rows), width="stretch", hide_index=True)

    st.subheader("原文与出处")
    for result in results:
        chunk = result.chunk
        with st.expander(
            f"#{result.rank} {chunk['company_name']}｜PDF第{chunk['pdf_page']}页｜{chunk['chunk_id']}"
        ):
            st.code(citation(chunk), language=None)
            st.write(chunk["text"])


if __name__ == "__main__":
    main()
