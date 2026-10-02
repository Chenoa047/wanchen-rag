# 万辰集团财报知识问答库

> 基于 10 家休闲食品上市公司 2025 年年报的本地财报 RAG 知识库，支持正文与表格解析、向量与 BM25 混合检索、跨公司分层召回、带 PDF 页码引用的回答以及检索效果评测。

本项目以万辰集团为核心，将万辰集团、三只松鼠、盐津铺子、劲仔食品、甘源食品、洽洽食品、好想你、良品铺子、来伊份和有友食品的年报正文与表格转换为可检索证据块。系统通过 Streamlit 展示答案、引用出处、BM25 排名、向量排名和 RRF 融合结果，使每个结论都能回到真实年报页面核查。

## 项目亮点

- **正文与表格并行解析**：使用 PyMuPDF 提取逐页正文，使用 pdfplumber 提取表格，保留表名、单位、表头和数据行。
- **可核查的引用**：每个知识块保存公司、章节、PDF 物理页码、内容类型和块 ID，回答可追溯至原始证据。
- **混合检索**：使用 Qwen3-Embedding-0.6B 向量检索和中文 BM25 检索，再通过 RRF 融合两类排名。
- **跨公司分层召回**：对“十家、排名、对比”类问题按公司保底召回，避免少数公司垄断上下文。
- **完整评测链路**：保存召回块、送入模型的上下文、回答全文、人工判定和错误类型，包括 Q8、Q9 的基线与分层召回对照。
- **安全与可复现**：原始 PDF、模型、索引和 API Key 不进入仓库，通过来源链接、文件名和 SHA-256 校验值保证语料可复现。

## 技术路线

```text
2025 年年报 PDF
    ↓
逐页正文提取 + 表格行列还原
    ↓
分页切块 + 公司/章节/页码元数据
    ↓
Qwen 向量索引 + jieba/BM25 索引
    ↓
RRF 融合 / 跨公司分层召回
    ↓
DeepSeek 基于证据生成带引用答案
    ↓
Streamlit 展示答案、排名、原文与出处
```

## 数据与评测结果

- [x] 确认 10 份 PDF 数量、公司、年份和全文属性
- [x] 记录 PDF 页数与 SHA-256 校验值
- [x] 用万辰集团年报验证正文、表格、章节和页码提取
- [x] 建立万辰单文件向量、BM25 与 RRF 检索，并通过 Q1-Q3 必要证据页 top-8 评测
- [x] 扩展至另外九家公司，完成 2,044 页、6,800 个检索块的自动验收与 20 页人工抽查
- [x] 完成 Streamlit 页面与无密钥启动测试
- [x] 完成 10 道正式题及附加题的真实回答、人工评价和一页结论
- [x] 完成 Q8、Q9 普通 RRF top-8 与每公司保底 2 块的分层召回对照

语料规模为 **10 家公司、2,044 个 PDF 页面、6,800 个检索块**。正式采用的 11 道题中，**8 道正确、3 道部分正确、0 道错误**。Q8、Q9 的对照实验表明：普通 RRF top-8 难以完整覆盖 10 家公司，而按公司分层召回能显著改善跨公司全景问题的证据覆盖。

原始年报由用户保存在 `/Users/DELL/Desktop/For Claude/年报`。PDF 不复制进 Git；项目通过 `data_manifest.csv` 记录文件位置和校验值。

## 环境安装

本项目要求 Python 3.12。在项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

## 项目结构

```text
wanchen-rag/
├── app.py                 # Streamlit 问答页面
├── config/                # jieba 财报词典
├── data_manifest.csv      # 年报清单、来源和校验值
├── evaluation/            # 题目、召回结果、模型回答和结论
├── screenshots/           # 单公司与跨公司页面截图
├── scripts/               # 提取、建库、评测和验收脚本
├── src/wanchen_rag/        # 提取、向量、BM25、RRF 和回答核心代码
└── tests/                 # 自动化测试
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

### 单公司问答

![万辰集团门店数问答](screenshots/02_single_company_retrieval.png)

### 跨公司问答与召回过程

![十家公司营收增速排名](screenshots/03_cross_company_retrieval.png)

![跨公司检索过程](screenshots/04_cross_company_retrieval_process.png)

## 局限性

- 财务表格存在“元、千元、万元”及合并/母公司口径差异，生成答案前仍需核对单位和表头。
- 跨公司问题需要更多上下文，分层召回可以提高覆盖率，但不能保证每个口径都完全一致。
- 当召回证据不足时，系统要求模型明确说明材料未覆盖，而不是使用常识补全。

## 项目标签

`RAG` `financial-reports` `knowledge-base` `hybrid-search` `BM25` `Qwen-Embedding` `Streamlit` `Python` `annual-report` `Chinese-NLP`

## 安全约束

DeepSeek 密钥由用户本人通过 `DEEPSEEK_API_KEY` 配置。项目代码、日志、页面和截图不得保存或显示密钥值。
