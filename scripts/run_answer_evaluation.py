from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wanchen_rag.answering import citation, generate_answer  # noqa: E402
from wanchen_rag.bm25 import load_index  # noqa: E402
from wanchen_rag.embedding import VectorIndex  # noqa: E402
from wanchen_rag.hybrid import HybridIndex  # noqa: E402
from wanchen_rag.evaluation import generation_succeeded  # noqa: E402


MAX_API_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5
RETRYABLE_API_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
ROUTE_ORDER = {
    "单公司 RRF top-8": 0,
    "RRF top-8 基线": 0,
    "分层召回，每公司2块": 1,
    "指定公司分层召回，每公司2块": 0,
}


def record_key(case_id: str, route: str) -> str:
    return f"{case_id}|{route}"


def routes_for_scope(scope: str) -> tuple[str, ...]:
    if scope == "all":
        return ("RRF top-8 基线", "分层召回，每公司2块")
    if "," in scope:
        return ("指定公司分层召回，每公司2块",)
    return ("单公司 RRF top-8",)


def load_saved_records(output_path: Path) -> dict[str, dict[str, object]]:
    if not output_path.exists():
        return {}
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(saved, list):
        raise ValueError("已有结果文件不是 JSON 列表")
    return {
        record_key(str(record["case_id"]), str(record.get("route", ""))): record
        for record in saved
        if isinstance(record, dict) and "case_id" in record
    }


def save_records(
    output_path: Path,
    records_by_key: dict[str, dict[str, object]],
    questions: list[dict[str, object]],
) -> None:
    question_order = {
        str(question["case_id"]): index for index, question in enumerate(questions)
    }
    ordered = sorted(
        records_by_key.values(),
        key=lambda record: (
            question_order.get(str(record["case_id"]), len(question_order)),
            ROUTE_ORDER.get(str(record.get("route", "")), 99),
        ),
    )
    output_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def request_answer_with_retry(
    question: str,
    results: list[object],
    *,
    api_key: str,
    model: str,
) -> str:
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        print(f"  API 请求 {attempt}/{MAX_API_ATTEMPTS}", flush=True)
        try:
            return generate_answer(
                question,
                results,
                api_key=api_key,
                model=model,
            )
        except RETRYABLE_API_ERRORS:
            if attempt == MAX_API_ATTEMPTS:
                raise
            print(
                f"  接口超时或暂时不可用，{RETRY_WAIT_SECONDS} 秒后自动重试……",
                flush=True,
            )
            time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError("未能生成回答")


def main() -> int:
    with (PROJECT_ROOT / "evaluation" / "questions.jsonl").open(encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream]
    output_path = PROJECT_ROOT / "evaluation" / "model_answers.json"
    try:
        records_by_key = load_saved_records(output_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"无法读取已有结果：{exc}")
        return 3
    if all(
        generation_succeeded(records_by_key.get(record_key(str(question["case_id"]), route)))
        for question in questions for route in routes_for_scope(str(question["scope"]))
    ):
        print("全部回答路线已有成功结果，无需加载模型或调用 API。")
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("未检测到 DEEPSEEK_API_KEY。请由您本人在当前终端配置后重试。")
        return 2
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

    with (PROJECT_ROOT / "data_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        all_stock_codes = [row["stock_code"] for row in csv.DictReader(stream)]

    index_dir = PROJECT_ROOT / "data" / "index"
    hybrid = HybridIndex(
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

    for question in questions:
        case_id = str(question["case_id"])
        scope = str(question["scope"])
        routes = routes_for_scope(scope)

        for route in routes:
            key = record_key(case_id, route)
            existing = records_by_key.get(key)
            if generation_succeeded(existing):
                print(f"{case_id} {route} 已有成功结果，跳过。", flush=True)
                continue

            if route == "RRF top-8 基线":
                results = hybrid.search(
                    str(question["question"]), top_k=8, candidate_k=20
                )
            elif scope == "all":
                results = hybrid.stratified_search(
                    str(question["question"]),
                    stock_codes=all_stock_codes,
                    per_company=2,
                    candidate_k=20,
                )
            elif "," in scope:
                results = hybrid.stratified_search(
                    str(question["question"]),
                    stock_codes=scope.split(","),
                    per_company=2,
                    candidate_k=20,
                )
            else:
                results = hybrid.search(
                    str(question["question"]),
                    top_k=8,
                    candidate_k=20,
                    stock_code=scope,
                )

            print(f"正在生成 {case_id} {route}……", flush=True)
            error_type = ""
            manual_evaluation = "待评价"
            try:
                answer = request_answer_with_retry(
                    str(question["question"]),
                    results,
                    api_key=api_key,
                    model=model,
                )
            except (AuthenticationError, PermissionDeniedError):
                print("API 身份验证或权限检查失败，请您本人检查密钥后重试。")
                return 4
            except RETRYABLE_API_ERRORS as exc:
                error_type = type(exc).__name__
                manual_evaluation = "待重新生成"
                answer = "API 请求在 3 次尝试后仍失败，本题未生成回答。"
                print(f"  {case_id} {route} 已记录失败，继续下一条。", flush=True)

            records_by_key[key] = {
                "case_id": case_id,
                "question": question["question"],
                "route": route,
                "model": model,
                "answer": answer,
                "sources": [citation(result.chunk) for result in results],
                "chunk_ids": [result.chunk["chunk_id"] for result in results],
                "manual_evaluation": manual_evaluation,
                "generation_status": "failed" if error_type else "success",
                "generation_error": error_type,
                "error_type": "",
            }
            save_records(output_path, records_by_key, questions)

    print(f"已保存 {len(records_by_key)} 条回答路线：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
