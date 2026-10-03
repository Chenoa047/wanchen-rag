"""将既有评测导出为可直接阅读的报告；不调用模型或读取凭据。"""
from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export_submission(root: Path) -> None:
    evaluation = root / "evaluation"
    answers = json.loads((evaluation / "model_answers.json").read_text(encoding="utf-8"))
    with (root / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reports = {row["stock_code"]: row for row in csv.DictReader(stream)}
    wanted = {chunk_id for answer in answers for chunk_id in answer["chunk_ids"]}
    chunks = {}
    with (root / "data/processed/all_chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            chunk = json.loads(line)
            if chunk["chunk_id"] in wanted:
                report = reports[chunk["stock_code"]]
                chunks[chunk["chunk_id"]] = {
                    **chunk,
                    "source_pdf": report["file_name"],
                    "source_url": report["source_url"],
                    "source_sha256": report["sha256"],
                }
    missing = wanted.difference(chunks)
    if missing:
        raise ValueError(f"回答引用的块不存在，停止导出：{sorted(missing)}")

    provenance = (
        "以下证据按原回答文件保存的 chunk_ids 从当前索引语料回查导出，"
        "不是当时 API 请求的完整日志。回答、人工评分和错误原因保持原样；"
        "原有检索评测查询与回答查询可能不同，未记录的历史排名不补造。"
    )
    review = [
        "# 评测结果", "",
        "本页汇总各题的模型回答、人工评价、错误分析及对应证据。", "",
        "[返回项目首页](../README.md) · [页面截图](../README.md#页面截图) · "
        "[评测结论](ONE_PAGE_CONCLUSION.md) · [原始评测汇总](results.csv)", "",
        "10 家公司，10 道正式题（Q1–Q10），Q11 为附加单位陷阱题。"
        "Q8、Q9 各有基线和分层召回两条路线，因此共有 13 条回答记录。", "",
        provenance, "",
        "## 题目目录", "",
    ]
    for index, answer in enumerate(answers, 1):
        review.append(f"- [{answer['case_id']}｜{answer['route']}](#r{index:02}) — {answer['question']}")
    for index, answer in enumerate(answers, 1):
        review.extend([
            "", f"## R{index:02}", "",
            f"**{answer['case_id']}｜{answer['route']}**", "", str(answer["question"]), "",
            f"人工评价：{answer.get('manual_evaluation', '待评价')}。"
            f"错误类别：{answer.get('error_type') or '无已标注错误'}。", "",
            str(answer.get("evaluation_note", "")), "",
            "### 保存的模型回答（原文）", "", str(answer["answer"]), "",
            "### 本条回答记录的证据块", "",
        ])
        for chunk_id in answer["chunk_ids"]:
            chunk = chunks[chunk_id]
            review.append(
                f"- [{chunk['company_name']}｜PDF第{chunk['pdf_page']}页｜{chunk_id}]"
                f"(EVIDENCE.md#{chunk_id})"
            )
    evidence = ["# 回答证据原文", "", provenance, "", "[返回逐题答案](REVIEW.md)", ""]
    for chunk_id, chunk in sorted(chunks.items()):
        evidence.extend([
            f"## {chunk_id}", "",
            f"{chunk['company_name']}｜{chunk['report_year']}年报｜PDF第{chunk['pdf_page']}页｜{chunk['chapter']}", "",
            f"[官方年报来源]({chunk['source_url']})；来源文件：`{chunk['source_pdf']}`。", "",
            *["> " + line for line in str(chunk["text"]).splitlines()], "",
        ])
    (evaluation / "REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    (evaluation / "EVIDENCE.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    (evaluation / "answer_evidence.json").write_text(
        json.dumps({"provenance": provenance, "chunks": chunks}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已导出 {len(answers)} 条原始回答、{len(chunks)} 个去重证据块。")


if __name__ == "__main__":
    export_submission(PROJECT_ROOT)
