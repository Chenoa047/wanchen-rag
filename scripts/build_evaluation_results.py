from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wanchen_rag.evaluation import generation_succeeded, evaluation_matches  # noqa: E402

EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def answer_route(case_id: str, route: str) -> str:
    if route == "RRF top-8":
        return "单公司 RRF top-8"
    if route == "分层召回":
        if case_id == "Q10":
            return "指定公司分层召回，每公司2块"
        return "分层召回，每公司2块"
    return route


def main() -> int:
    rows: list[dict[str, str]] = []

    local_records = json.loads(
        (EVALUATION_DIR / "wanchen_retrieval_local_questions.json").read_text(
            encoding="utf-8"
        )
    )
    for record in local_records:
        passed = record["status"] == "PASS"
        rows.append(
            {
                "case_id": record["case_id"],
                "route": "RRF top-8",
                "question": record["query"],
                "retrieval_status": record["status"],
                "company_coverage": "万辰集团",
                "answer_status": "待用户本人配置 DeepSeek API 后运行",
                "evaluation": "仅完成召回评测",
                "error_type": "" if passed else "未召回必要证据页",
            }
        )

    cross_records = json.loads(
        (EVALUATION_DIR / "cross_company_retrieval.json").read_text(encoding="utf-8")
    )
    for record in cross_records:
        targets = {str(value) for value in record["target_stock_codes"]}
        for route_key, route_label in (
            ("baseline", "RRF top-8 基线"),
            ("stratified", "分层召回"),
        ):
            route = record[route_key]
            coverage_key = (
                "target_company_coverage" if route_key == "baseline" else "company_coverage"
            )
            covered = {str(value) for value in route[coverage_key]}
            passed = targets.issubset(covered)
            rows.append(
                {
                    "case_id": record["case_id"],
                    "route": route_label,
                    "question": record["query"],
                    "retrieval_status": "PASS" if passed else "FAIL",
                    "company_coverage": f"{len(covered)}/{len(targets)}",
                    "answer_status": "待用户本人配置 DeepSeek API 后运行",
                    "evaluation": "仅完成召回评测",
                    "error_type": "" if passed else "跨公司覆盖不足",
                }
            )

    model_answers = json.loads(
        (EVALUATION_DIR / "model_answers.json").read_text(encoding="utf-8")
    )
    answer_keys = {
        (str(record["case_id"]), str(record["route"])) for record in model_answers
        if generation_succeeded(record)
    }
    manual_evaluations = json.loads(
        (EVALUATION_DIR / "manual_evaluations.json").read_text(encoding="utf-8")
    )
    evaluations_by_key = {
        (str(record["case_id"]), str(record["route"])): record
        for record in manual_evaluations
    }
    valid_evaluations = {}

    for answer in model_answers:
        key = (str(answer["case_id"]), str(answer["route"]))
        evaluation = evaluations_by_key.get(key)
        if evaluation_matches(answer, evaluation):
            valid_evaluations[key] = evaluation
            answer["manual_evaluation"] = evaluation["answer_status"]
            answer["evaluation_note"] = evaluation["evaluation"]
            answer["error_type"] = evaluation["error_type"]
    (EVALUATION_DIR / "model_answers.json").write_text(
        json.dumps(model_answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for row in rows:
        key = (row["case_id"], answer_route(row["case_id"], row["route"]))
        evaluation = valid_evaluations.get(key)
        if evaluation and key in answer_keys:
            row["answer_status"] = str(evaluation["answer_status"])
            row["evaluation"] = str(evaluation["evaluation"])
            row["error_type"] = str(evaluation["error_type"])
        elif key in answer_keys:
            row["answer_status"] = "已生成，待人工评价"
            row["evaluation"] = "待评价"
        else:
            row["answer_status"] = "无成功回答"
            row["evaluation"] = "仅召回对照或接口请求失败，不能套用历史人工评价"

    output_path = EVALUATION_DIR / "results.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"评测汇总：{output_path}，共 {len(rows)} 条路线记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
