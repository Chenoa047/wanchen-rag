# 万辰集团财报知识问答库

本项目使用 10 家休闲食品相关 A 股公司的 2025 年年度报告全文，构建本地混合检索与带出处问答页面。

## 当前进度

- [x] 确认 10 份 PDF 数量、公司、年份和全文属性
- [x] 记录 PDF 页数与 SHA-256 校验值
- [x] 用万辰集团年报验证正文、表格、章节和页码提取
- [x] 建立万辰单文件向量、BM25 与 RRF 检索，并通过 Q1-Q3 必要证据页 top-8 评测
- [x] 扩展至另外九家公司，完成 2,044 页、6,800 个检索块的自动验收与 20 页人工抽查
- [x] 完成 Streamlit 页面与无密钥启动测试
- [x] 完成 10 道正式题及附加题的真实回答、人工评价和一页结论
- [x] 完成 Q8、Q9 普通 RRF top-8 与每公司保底 2 块的分层召回对照

原始年报由用户保存在 `/Users/DELL/Desktop/For Claude/年报`。PDF 不复制进 Git；项目通过 `data_manifest.csv` 记录文件位置和校验值。

## 环境安装

本项目要求 Python 3.12。在项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

## 建库、评测与启动

```bash
.venv/bin/python scripts/validate_reports.py
.venv/bin/python scripts/extract_wanchen.py
.venv/bin/python scripts/validate_wanchen_extraction.py
.venv/bin/python scripts/extract_all_reports.py
.venv/bin/python scripts/validate_all_extraction.py
.venv/bin/python scripts/build_full_corpus.py
.venv/bin/python scripts/build_all_bm25.py
.venv/bin/python scripts/build_all_vectors.py
.venv/bin/python scripts/evaluate_cross_company_retrieval.py
.venv/bin/python scripts/build_financial_rankings.py
.venv/bin/python scripts/build_evaluation_results.py
.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
.venv/bin/python scripts/build_wanchen_bm25.py
.venv/bin/python scripts/evaluate_wanchen_bm25.py
.venv/bin/python scripts/build_wanchen_vectors.py
.venv/bin/python scripts/evaluate_wanchen_vectors.py
.venv/bin/python scripts/evaluate_wanchen_hybrid.py
.venv/bin/python scripts/prepare_wanchen_answer.py "万辰集团2025年末门店数量是多少？"
.venv/bin/python -m unittest discover -s tests -v
```

提取结果保存在 `data/processed/300972/`，其中：

- `pages.jsonl`：逐页清洗后的正文；
- `tables.jsonl`：表格原始行列、表名、单位和页码；
- `chunks.jsonl`：供后续 BM25 与向量索引使用的正文块、表格块。

全库自动验收与 20 页人工抽查记录见 `evaluation/CORPUS_QA.md`。
十道正式题和一道单位陷阱附加题见 `evaluation/questions.jsonl`，人工参考答案见 `evaluation/WANCHEN_Q1_Q7_REFERENCE.md`、`evaluation/Q8_Q9_REFERENCE.md` 和 `evaluation/Q10_REFERENCE.md`。

页面在未配置密钥时仍可执行本地混合检索并展示原文出处；输入已完成评测的题目时，还会回放 `evaluation/model_answers.json` 中已保存的带引用答案。需要生成新答案时，请由用户本人在启动命令所在的终端配置 `DEEPSEEK_API_KEY`。默认模型为 `deepseek-flash`；如需更换，可配置非秘密变量 `DEEPSEEK_MODEL`。项目不保存、显示或记录密钥。
如需重新运行全部 10 道正式题和附加题，由用户本人在已配置密钥的终端执行 `.venv/bin/python scripts/run_answer_evaluation.py`。脚本会断点续跑，并为 Q8、Q9 分别保存普通 RRF top-8 基线和分层召回答案，共形成 13 条回答路线记录。该命令会产生 API 费用，结果写入 `evaluation/model_answers.json`，不会写入密钥。

`evaluation/results.csv` 保存 14 条召回/回答路线的覆盖情况、正确性判定和错误类别。按正式采用的 11 道题路线统计，8 道正确，Q6、Q7、Q10 为部分正确。`data_manifest.csv` 已补齐 10 份年报的披露日期和官方来源网址。

## 页面截图

- `screenshots/01_corpus_stats.png`：10 家公司的年报页数和检索块统计。
- `screenshots/02_single_company_retrieval.png`：万辰集团门店数问题的带引用答案。
- `screenshots/03_cross_company_retrieval.png`：十家公司营收增速排名的分层召回答案。
- `screenshots/04_cross_company_retrieval_process.png`：跨公司问题的 BM25、向量与 RRF 排名过程。
- `screenshots/05_single_company_retrieval_process.png`：单公司问题的召回排名和原文出处入口。

## 安全约束

DeepSeek 密钥由用户本人通过 `DEEPSEEK_API_KEY` 配置。项目代码、日志、页面和截图不得保存或显示密钥值。
