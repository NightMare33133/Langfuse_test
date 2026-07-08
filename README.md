# Langfuse RAG 评测工具

基于 Langfuse 导出数据的 RAG 检索 + 回答质量自动评测工具。支持 JSONL 解析、LLM Judge 评分、可视化看板与报告导出。

## 功能概览

- **数据导入** — 支持手动上传 JSONL 文件或直接从 Langfuse API 拉取 Traces
- **数据解析** — 将 Langfuse 导出的 JSONL（observation/span 级别）按 `traceId` 聚合为结构化样本
- **LLM Judge 评测** — 调用 OpenAI 兼容 API 对每条样本自动评分（Top1/Top3/Top5 命中 + 回答正确性）
- **评测优化** — 规则预筛选（无检索/无回答直接判定）+ 内容级去重，大幅减少 LLM 调用次数
- **可视化看板** — 指标卡片、柱状图、饼图、每题命中热力图
- **Top1 未命中分析** — 快速定位 Top1 未命中但 Top3 命中的案例
- **报告导出** — 一键下载 CSV 或 Markdown 评测报告

## 项目结构

```
Langfuse_test/
├── app.py                    # Streamlit 主界面
├── judge.py                  # LLM Judge 模块（prompt 构建、API 调用、指标计算、预筛选与去重）
├── parser.py                 # Langfuse JSONL 解析模块
├── fetch_traces.py           # Langfuse API Trace 拉取模块
├── main.py                   # CLI 入口（仅解析，不含 Judge）
├── question.py               # Dify 批量提问脚本
├── prompts/
│   └── judge_prompt.txt      # Judge Prompt 模板
├── data/
│   ├── raw/                  # 上传或拉取的 Langfuse 原始 JSONL
│   ├── processed/            # 解析后的样本 + 摘要
│   │   ├── langfuse_samples.jsonl
│   │   └── langfuse_summary.json
│   └── judged/               # Judge 评测结果
│       ├── eval_results.jsonl
│       └── eval_results_<时间戳>.jsonl   # 历史快照
├── .env.example              # 环境变量模板
└── README.md
```

## 环境要求

- Python 3.13+
- 依赖：`streamlit`、`pandas`、`plotly`、`requests`、`python-dotenv`

```bash
pip install streamlit pandas plotly requests python-dotenv
```

## 配置

复制 `.env.example` 为 `.env` 并填写对应配置：

```bash
cp .env.example .env
```

### 环境变量说明

| 变量 | 说明 |
|------|------|
| `LANGFUSE_HOST` | Langfuse 服务地址（默认 `http://localhost:3000`） |
| `LANGFUSE_PUBLIC_KEY` | Langfuse Public Key（用于 API 拉取 Traces） |
| `LANGFUSE_SECRET_KEY` | Langfuse Secret Key |
| `JUDGE_API_KEY` | Judge LLM 的 API Key |
| `JUDGE_API_BASE` | Judge LLM 的 Base URL（默认 `https://token-plan-cn.xiaomimimo.com/v1`） |
| `JUDGE_MODEL` | Judge 使用的模型名称（默认 `mimo-v2.5-pro`） |
| `DIFY_API_KEY` | Dify API Key（用于 `question.py` 批量提问） |

## 使用方法

### 1. 启动 Streamlit 应用

```bash
streamlit run app.py
```

### 2. 数据导入

支持两种方式导入数据：

#### 方式 A：上传文件

1. 在左侧边栏「数据导入」区域上传 Langfuse 导出的 `.jsonl` 文件
2. 文件会自动保存到 `data/raw/` 目录

#### 方式 B：从 Langfuse API 拉取

1. 在左侧边栏填写 Langfuse 地址、Public Key、Secret Key
2. 设置每页拉取的 trace 数量
3. 点击「拉取 Traces」，数据会自动保存为带时间戳的 `.jsonl` 文件到 `data/raw/`

也可通过 CLI 独立拉取：

```bash
python fetch_traces.py --host http://localhost:3000 --public-key <PK> --secret-key <SK> --limit 50 --max-pages 20
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--host` | 否 | Langfuse 服务地址（默认读 `.env`） |
| `--public-key` | 否 | Langfuse Public Key（默认读 `.env`） |
| `--secret-key` | 否 | Langfuse Secret Key（默认读 `.env`） |
| `--limit` | 否 | 每页 trace 数量，默认 50 |
| `--max-pages` | 否 | 最大翻页数，默认 20 |
| `--output` | 否 | 输出 JSONL 路径，默认 `data/raw/langfuse_api_export_<时间戳>.jsonl` |

### 3. 数据解析

1. 在左侧边栏选择 `data/raw/` 下已有文件
2. 点击「开始解析」
3. 解析结果保存到 `data/processed/`，切换到「样本列表」tab 查看

### 4. Judge 评测

1. 在左侧边栏填写 API Key、Base URL、Model
2. 可先点击「测试 Judge 连接」验证配置
3. 选择评测范围：勾选「只评前 1 条」快速试跑，或设置批量评测数量
4. 可选：点击「预览优化策略」查看实际需要调用 LLM 的次数（不会消耗 token）
5. 点击「运行 Judge 评测」
6. 切换到「评测结果」tab 查看实时进度和最终结果

#### 评测优化机制

- **规则预筛选**：无问题、无检索结果、无最终回答的样本由规则直接判定，不调用 LLM
- **内容去重**：相同 `question + retrieval_query + final_answer` 的样本只评一次，复用首次结果
- **跳过已有结果**：已有成功评测的样本不会重复调用 LLM（可取消勾选或强制重跑）
- **重试失败样本**：可单独重试之前评测失败的样本
- **Prompt 裁剪**：检索结果正文超过 300 字符时自动截断，减少 token 消耗

#### 高级选项

- **显示 Judge Prompt 和原始响应**：调试模式，展示每条样本的 prompt 和 LLM 原始返回
- **强制重新评测**：忽略所有缓存，重新评测全部样本

### 5. CLI 模式（仅解析）

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
