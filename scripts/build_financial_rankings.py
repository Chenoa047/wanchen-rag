from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "evaluation" / "financial_snapshot.csv"
OUTPUT = PROJECT_ROOT / "evaluation" / "Q8_Q9_REFERENCE.md"


def money_yi(value: Decimal) -> str:
    return f"{value / Decimal('100000000'):.2f} 亿元"


def citation(row: dict[str, str]) -> str:
    return (
        f"[{row['company_name']}｜2025年报｜PDF第{row['pdf_page']}页｜"
        f"{row['source_chunk_id']}]"
    )


def main() -> int:
    with SNAPSHOT.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 10:
        raise AssertionError(f"财务快照应为 10 家，实际 {len(rows)} 家")
    for row in rows:
        revenue_2025 = Decimal(row["revenue_2025_yuan"])
        revenue_2024 = Decimal(row["revenue_2024_yuan"])
        calculated = (revenue_2025 / revenue_2024 - 1) * 100
        reported = Decimal(row["revenue_yoy_pct"])
        if abs(calculated - reported) > Decimal("0.02"):
            raise AssertionError(
                f"{row['company_name']}营收增速校验失败：计算{calculated:.4f}%，报告{reported}%"
            )

    revenue_rank = sorted(rows, key=lambda row: Decimal(row["revenue_2025_yuan"]), reverse=True)
    profit_rank = sorted(rows, key=lambda row: Decimal(row["net_profit_2025_yuan"]), reverse=True)
    growth_rank = sorted(rows, key=lambda row: Decimal(row["revenue_yoy_pct"]), reverse=True)

    lines = [
        "# Q8/Q9 人工参考答案",
        "",
        "所有金额与同比均来自各公司 2025 年年度报告的主要会计数据表。金额排名使用原始人民币元计算，表中换算为亿元仅为便于阅读。",
        "",
        "## Q8 营业收入排名",
        "",
        "| 排名 | 公司 | 2025 年营业收入 | 同比 | 出处 |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(revenue_rank, start=1):
        lines.append(
            f"| {rank} | {row['company_name']} | {money_yi(Decimal(row['revenue_2025_yuan']))} | "
            f"{Decimal(row['revenue_yoy_pct']):.2f}% | {citation(row)} |"
        )

    lines.extend(
        [
            "",
            "## Q8 归母净利润排名",
            "",
            "| 排名 | 公司 | 2025 年归母净利润 | 报告同比 | 出处 |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(profit_rank, start=1):
        yoy = f"{Decimal(row['net_profit_yoy_pct']):.2f}%" if row["net_profit_yoy_pct"] else "不适用"
        lines.append(
            f"| {rank} | {row['company_name']} | {money_yi(Decimal(row['net_profit_2025_yuan']))} | "
            f"{yoy} | {citation(row)} |"
        )

    lines.extend(
        [
            "",
            "## Q9 营业收入同比增速排名",
            "",
            "| 排名 | 公司 | 营收同比 | 出处 |",
            "|---:|---|---:|---|",
        ]
    )
    for rank, row in enumerate(growth_rank, start=1):
        lines.append(
            f"| {rank} | {row['company_name']} | {Decimal(row['revenue_yoy_pct']):.2f}% | {citation(row)} |"
        )
    declining = "、".join(row["company_name"] for row in growth_rank if Decimal(row["revenue_yoy_pct"]) < 0)
    lines.extend(
        [
            "",
            f"增长最快：{growth_rank[0]['company_name']}（{Decimal(growth_rank[0]['revenue_yoy_pct']):.2f}%）。",
            f"出现下滑：{declining}。",
            "",
            "口径提醒：好想你 2024 年数据采用报告中的会计政策变更后调整口径；良品铺子 2025 年归母净利润为负，报告将其同比列为“不适用”。",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS：10 家营收增速均通过反算校验，参考答案已写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

