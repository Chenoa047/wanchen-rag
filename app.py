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


@st.cache_data
def load_manifest() -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_corpus_stats() -> list[dict[str, object]]:
    rows = []
    for report in load_manifest():
        chunks_path = (
            PROJECT_ROOT / "data" / "processed" / report["stock_code"] / "chunks.jsonl"
        )
        chunk_count = None
        if chunks_path.is_file():
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


def load_saved_answers() -> list[dict[str, object]]:
    answers_path = PROJECT_ROOT / "evaluation" / "model_answers.json"
    if not answers_path.exists():
        return []
    return json.loads(answers_path.read_text(encoding="utf-8"))


def show_saved_evaluation() -> None:
    records = load_saved_answers()
    if not records:
        st.info("尚无已保存的评测答案，请先完成评测。")
        return
    selected = st.selectbox(
        "已保存题目与路线", range(len(records)),
        format_func=lambda index: f"{records[index]['case_id']}｜{records[index]['route']}",
        key="saved_record",
    )
    record = records[selected]
    st.caption("历史评测回放：不调用 API，不加载本地模型；不是重新生成的答案。")
    st.subheader(str(record["question"]))
    st.markdown(str(record["answer"]))
    st.caption(f"人工评价：{record.get('manual_evaluation', '待评价')}")
    if record.get("evaluation_note"):
        st.info(str(record["evaluation_note"]))
    with st.expander("本次回答保存的引用来源"):
        for source in record.get("sources", []):
            st.text(source)

    # 只展示本条回答实际记录的块，绝不与一次新的检索混搭。
    evidence = {}
    for filename in ("wanchen_retrieval_local_questions.json", "cross_company_retrieval.json"):
        path = PROJECT_ROOT / "evaluation" / filename
        if not path.is_file():
            continue
        for retrieval in json.loads(path.read_text(encoding="utf-8")):
            if retrieval["case_id"] != record["case_id"]:
                continue
            if "results" in retrieval:
                rows = retrieval["results"]
            else:
                route = "baseline" if record["route"] == "RRF top-8 基线" else "stratified"
                rows = retrieval[route]["results"]
            evidence.update({row["chunk_id"]: row for row in rows})
    st.subheader("已保存的召回证据")
    st.caption("按回答文件记录的上下文顺序展示；排名来自同题同路线的检索评测记录。")
    company_names = {row["stock_code"]: row["company_name"] for row in load_manifest()}
    snapshot_path = PROJECT_ROOT / "evaluation" / "answer_evidence.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        st.caption(snapshot["provenance"])
        for chunk_id, chunk in snapshot["chunks"].items():
            evidence.setdefault(chunk_id, chunk)
    matched = [evidence[chunk_id] for chunk_id in record.get("chunk_ids", []) if chunk_id in evidence]
    if len(matched) != len(record.get("chunk_ids", [])):
        st.warning("部分回答引用块未保存在对应检索记录中；此处只展示可核对的块，不补写证据。")
    st.dataframe(pd.DataFrame([
        {
            "块ID": row["chunk_id"],
            "公司": company_names.get(str(row["chunk_id"]).split("-")[0], "未知"),
            "PDF页": row["pdf_page"],
            "历史RRF排名": row.get("rrf_rank", row.get("rank")),
            "BM25排名": row.get("bm25_rank"),
            "向量排名": row.get("vector_rank"),
            "RRF得分": row.get("rrf_score"),
        }
        for row in matched
    ]), width="stretch", hide_index=True)
    for row in matched:
        with st.expander(f"PDF第{row['pdf_page']}页｜{row['chunk_id']}"):
            st.write(row["text"])


def missing_local_files() -> list[str]:
    required = (
        "data/processed/all_chunks.jsonl", "data/index/all_bm25.pkl",
        "data/index/all_embeddings.npy", "data/index/all_embeddings.json",
    )
    return [name for name in required if not (PROJECT_ROOT / name).is_file()]


@st.cache_resource
def load_hybrid_index():
    from wanchen_rag.bm25 import load_index
    from wanchen_rag.embedding import VectorIndex
    from wanchen_rag.hybrid import HybridIndex

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
    total_chunks = sum(int(row["检索块"]) for row in corpus_stats if row["检索块"] is not None)
    complete_corpus = all(row["检索块"] is not None for row in corpus_stats)

    with st.sidebar:
        st.header("检索设置")
        mode = st.radio("使用方式", ["查看已保存评测", "本地检索与提问"])
        scope = st.selectbox("公司范围", ["全行业", *company_to_code], key="scope")
        stratified = st.checkbox(
            "跨公司分层召回",
            value=scope == "全行业",
            disabled=scope != "全行业",
            help="每家公司至少召回 2 个证据块，防止单一公司垄断结果。",
        )
        st.metric("公司", len(manifest))
        st.metric("年报页数", f"{total_pages:,}")
        st.metric("本地检索块", f"{total_chunks:,}" if complete_corpus else "未完整建库")

    with st.expander("语料统计（按公司）"):
        st.dataframe(pd.DataFrame(corpus_stats), width="stretch", hide_index=True)

    if mode == "查看已保存评测":
        st.info("下方展示历史评测，题目自带公司范围，不受侧栏检索设置影响。")
        show_saved_evaluation()
        return
    missing = missing_local_files()
    if missing:
        st.warning("本地知识库尚未建好。请按 README 下载年报并完成建库；也可切回已保存评测。")
        st.code("\n".join(missing), language=None)
        return

    question = st.text_area(
        "请输入财报问题",
        placeholder="例如：十家公司 2025 年营业收入同比增速如何排名？",
        height=100,
    )
    submitted = st.button("检索证据", type="primary", disabled=not question.strip())
    context = (question.strip(), scope, stratified)
    if st.session_state.get("result_context") != context:
        st.session_state.pop("search_result", None)
    if not submitted and "search_result" not in st.session_state:
        st.info("无密钥时可检索证据；历史答案请在“查看已保存评测”中查看。")
        return
    if submitted:
        try:
            with st.spinner("正在加载本地模型并检索……"):
                hybrid = load_hybrid_index()
                if scope == "全行业" and stratified:
                    results = hybrid.stratified_search(
                        question, stock_codes=[row["stock_code"] for row in manifest],
                        per_company=2, candidate_k=20,
                    )
                else:
                    results = hybrid.search(question, top_k=8, candidate_k=20,
                                            stock_code=company_to_code.get(scope))
        except Exception:
            st.error("无法加载本地索引或模型。请按 README 完成建库，并确认模型缓存完整。")
            return
        answer = ""
        message = "未配置回答服务；以下为本次检索证据，历史答案请切换到已保存评测。"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            try:
                with st.spinner("正在基于召回材料生成带引用的答案……"):
                    answer = generate_answer(question, results, api_key=api_key,
                                             model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
            except Exception:
                message = "答案生成失败；本地检索结果仍可使用。请由您本人检查接口配置后重试。"
        st.session_state["result_context"] = context
        st.session_state["search_result"] = (results, answer, message)

    results, answer, message = st.session_state["search_result"]
    st.subheader("答案")
    if answer:
        st.markdown(answer)
    else:
        st.info(message)

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
