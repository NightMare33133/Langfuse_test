# Langfuse RAG 评测工具

基于 Langfuse 导出数据的 RAG 检索 + 回答质量自动评测工具。支持 JSONL 解析、LLM Judge 评分、可视化看板与报告导出。

## 功能概览

- **数据解析** — 将 Langfuse 导出的 JSONL（observation/span 级别）按 `traceId` 聚合为结构化样本
- **LLM Judge 评测** — 调用 OpenAI 兼容 API 对每条样本自动评分（Top1/Top3/Top5 命中 + 回答正确性）
- **可视化看板** — 指标卡片、柱状图、饼图、每题命中热力图
- **Top1 未命中分析** — 快速定位 Top1 未命中但 Top3 命中的案例
- **报告导出** — 一键下载 CSV 或 Markdown 评测报告

## 项目结构

```
Langfuse_test/
├── app.py                    # Streamlit 主界面
├── judge.py                  # LLM Judge 模块（prompt 构建、API 调用、指标计算）
├── parser.py                 # Langfuse JSONL 解析模块
├── main.py                   # CLI 入口（仅解析，不含 Judge）
├── prompts/
│   └── judge_prompt.txt      # Judge Prompt 模板
├── data/
│   ├── raw/                  # 上传的 Langfuse 原始 JSONL
│   ├── processed/            # 解析后的样本 + 摘要
│   │   ├── langfuse_samples.jsonl
│   │   └── langfuse_summary.json
│   └── judged/               # Judge 评测结果
│       └── eval_results.jsonl
└── README.md
```

## 环境要求

- Python 3.13+
- 依赖：`streamlit`、`pandas`、`plotly`、`requests`

```bash
pip install streamlit pandas plotly requests
```

## 使用方法

### 1. 启动 Streamlit 应用

```bash
streamlit run app.py
```

### 2. 数据导入与解析

1. 在左侧边栏上传 Langfuse 导出的 `.jsonl` 文件，或选择 `data/raw/` 下已有文件
2. 点击「开始解析」
3. 解析结果保存到 `data/processed/`，切换到「样本列表」tab 查看

### 3. Judge 评测

1. 在左侧边栏填写 API Key、Base URL、Model
2. 可先点击「测试 Judge 连接」验证配置
3. 勾选「只评测前 1 条样本」可快速试跑；取消勾选可设置批量评测数量
4. 点击「运行 Judge 评测」
5. 切换到「评测结果」tab 查看实时进度和最终结果

### 4. CLI 模式（仅解析）

```bash
python main.py <input.jsonl> [--output PATH] [--summary PATH]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `input` | 是 | Langfuse 导出的 `.jsonl` 文件路径 |
| `--output` | 否 | 输出 JSONL 路径，默认 `<input>.samples.jsonl` |
| `--summary` | 否 | 摘要 JSON 路径，默认 `<input>.summary.json` |

## 评测结果看板

评测完成后，「评测结果」tab 包含以下内容：

### 指标卡片

总样本数 | 有效评测数 | 错误数 | Top1 Hit | Top3 Hit | Top5 Hit | Answer Correctness

### 可视化图表

- **命中率柱状图** — Top1 / Top3 / Top5 / Answer Correctness 四项百分比
- **Answer 饼图** — 正确 vs 错误占比
- **每题命中图** — 按问题展示 Top1 / Top3 / Answer 命中情况，未命中案例排在前面

### Top1 未命中案例

筛选 `retrieval_top1_hit == 0` 的样本，显示问题、原因、Top3/Top5 状态，便于分析检索质量问题。

### 评测详情表格

完整数据表格，包含 question、各项 hit 指标、reason、trace_id，支持 Streamlit 内置排序和搜索。

### 导出

- **下载 CSV** — 评测结果的 CSV 文件（UTF-8 with BOM，Excel 友好）
- **下载 Markdown 报告** — 包含指标汇总、命中率表格、Top1 未命中列表、每题详情

## Judge 评分标准

Judge Prompt 指示 LLM 对每条样本输出以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `retrieval_top1_hit` | 0/1 | Top1 检索结果是否包含正确答案 |
| `retrieval_top3_hit` | 0/1 | Top3 检索结果中是否包含正确答案 |
| `retrieval_top5_hit` | 0/1 | Top5 检索结果中是否包含正确答案 |
| `answer_correct` | 0/1 | 最终回答是否正确 |
| `reason` | string | 评分理由（100 字以内） |

> **注意**：如果每题实际只召回 3 条检索结果，则 Top5 指标仅供参考，严格来说需要把 Dify 检索 topK 调到 5 后重新测试。

## 输出格式

### 样本 JSONL（`data/processed/langfuse_samples.jsonl`）

```json
{
  "trace_id": "abc123",
  "trace_name": "workflow_name",
  "session_id": "session_001",
  "user_id": "user_001",
  "workflow_run_id": "run_001",
  "question": "P2P借贷是什么？",
  "retrieval_query": "P2P借贷",
  "retrieval_results": [
    {
      "position": 1,
      "score": 0.95,
      "title": "P2P借贷简介",
      "content": "..."
    }
  ],
  "llm_model": "gpt-4",
  "llm_input": {},
  "llm_output": {},
  "final_answer": "P2P借贷是指...",
  "observations": []
}
```

### 评测结果 JSONL（`data/judged/eval_results.jsonl`）

```json
{
  "trace_id": "abc123",
  "question": "P2P借贷是什么？",
  "retrieval_top1_hit": 1,
  "retrieval_top3_hit": 1,
  "retrieval_top5_hit": 1,
  "answer_correct": 1,
  "reason": "Top1 检索结果包含相关内容，回答准确"
}
```
