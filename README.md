# Langfuse RAG 评测工具

基于 Langfuse 导出数据的 RAG 检索 + 回答质量自动评测工具。支持题目生成、批量提问、样本解析、参考答案回填、LLM Judge 评分、可视化看板与报告导出。

## 功能概览

- **题目生成** — 上传知识库文件，自动按章节切分后调用 LLM 生成带参考答案的测评题目
- **批量提问** — 将题目批量发送到 Dify Q&A 接口，收集回答和检索结果，参考答案自动透传
- **样本准备** — 导入 Langfuse / Dify 记录，解析为结构化样本，并从题目库回填 reference_answer
- **Judge 评测** — 严格评测（有参考答案）和合理性评测（无参考答案）两种模式，自动选择 prompt
- **评测优化** — 规则预筛选 + 内容级去重 + 分层 prompt 截断，大幅减少 LLM 调用次数
- **可视化看板** — 指标卡片、柱状图、饼图、每题命中热力图
- **报告导出** — 一键下载 CSV 或 Markdown 评测报告

## 四步工作流

```
题目生成 → 批量提问 → 样本准备 → Judge 评测
```

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1. 题目生成 | `question_generator.py` | 上传知识库文件，调用 LLM 生成题目，输出 question / reference_answer / source_excerpt / difficulty / topic |
| 2. 批量提问 | `batch_query.py` | 将题目逐条发送到 Dify，收集 final_answer 和 retrieval_results，参考答案透传 |
| 3. 样本准备 | `parser.py` | 解析 Langfuse / Dify 记录为结构化样本，从题目库回填 reference_answer |
| 4. Judge 评测 | `judge.py` + `app.py` | 配置 API，选择评测范围，调用 LLM 评分，查看结果和可视化 |

### 评测模式

| 模式 | 触发条件 | Answer Correct 含义 |
|------|---------|-------------------|
| **严格评测** | 样本有 `reference_answer` | 回答是否与参考答案一致、覆盖关键要点 |
| **合理性评测** | 样本无 `reference_answer` | 回答是否基于检索内容看起来合理、完整 |

### 参考答案回填

样本准备阶段会自动从题目库（`data/questions/*.jsonl`）中匹配并回填：

1. 按 `question_id` 精确匹配（优先）
2. 按 `question` 文本精确匹配（兜底）
3. 匹配成功 → 回填 reference_answer、source_excerpt、difficulty、topic
4. 匹配失败 → 保留为空，走无参考答案评测

## 项目结构

```
Langfuse_test/
├── app.py                    # Streamlit 主界面
├── judge.py                  # LLM Judge 模块（prompt 构建、API 调用、指标计算）
├── parser.py                 # Langfuse JSONL 解析 + 参考答案回填
├── question_generator.py     # 题目生成模块
├── batch_query.py            # 批量提问模块
├── fetch_traces.py           # Langfuse API Trace 拉取模块
├── main.py                   # CLI 入口（仅解析，不含 Judge）
├── prompts/
│   ├── judge_prompt.txt      # Judge Prompt（无参考答案模板）
│   ├── judge_prompt_with_ref.txt  # Judge Prompt（有参考答案模板）
│   └── qgen_prompt.txt       # 题目生成 Prompt
├── data/
│   ├── raw/                  # 上传或拉取的原始 JSONL
│   ├── processed/            # 解析后的结构化样本
│   │   ├── langfuse_samples.jsonl
│   │   └── langfuse_summary.json
│   ├── judged/               # Judge 评测结果
│   │   ├── eval_results.jsonl
│   │   └── eval_results_<时间戳>.jsonl   # 历史快照
│   ├── questions/            # 生成的题目文件
│   └── batch/                # 批量提问结果
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
| `DIFY_API_KEY` | Dify API Key（用于批量提问） |

## 使用方法

### 1. 启动 Streamlit 应用

```bash
 streamlit run app.py
```

### 2. 题目生成

1. 在「题目生成」tab 上传知识库文件（.txt / .md）
2. 设置生成数量、难度偏好、生成策略
3. 点击「生成题目」
4. 题目保存到 `data/questions/`，包含 question、reference_answer、source_excerpt、difficulty、topic

### 3. 批量提问

1. 在「批量提问」tab 选择问题来源（已生成题目 / 手动输入 / 文件加载）
2. 配置 Dify API Key 和地址
3. 点击「开始提问」
4. 成功结果可推送到 `data/raw/`，供样本准备解析

### 4. 样本准备

1. 在「样本准备」tab 上传 Langfuse JSONL 或从 API 拉取
2. 选择文件并点击「开始解析」
3. 解析结果保存到 `data/processed/langfuse_samples.jsonl`
4. 自动从题目库回填 reference_answer，解析后显示回填统计

### 5. Judge 评测

1. 在「Judge 评测」tab 配置 API（Key、Base URL、Model）
2. 选择评测范围和模式（跳过已有结果 / 强制重评）
3. 可点击「预览优化策略」查看实际 LLM 调用次数
4. 点击「运行 Judge 评测」
5. 查看指标、可视化图表、评测详情

### 6. CLI 模式（仅解析）

```bash
python main.py <input.jsonl> [--output PATH] [--summary PATH]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `input` | 是 | Langfuse 导出的 `.jsonl` 文件路径 |
| `--output` | 否 | 输出 JSONL 路径，默认 `<input>.samples.jsonl` |
| `--summary` | 否 | 摘要 JSON 路径，默认 `<input>.summary.json` |

## 评测结果看板

评测完成后，「Judge 评测」tab 包含以下内容：

### 指标卡片

总样本数 | 有效评测数 | 错误数 | Top1 Hit | Top3 Hit | Top5 Hit | Answer Correctness

### 评测模式说明

- **纯严格评测**：全部样本均有参考答案，Answer OK = 与参考答案对比的正确性
- **纯合理性评测**：全部样本无参考答案，Answer OK = 基于检索内容判断的合理性
- **混合模式**：两种口径混合，总正确率需谨慎解读

### 可视化图表

- **命中率柱状图** — Top1 / Top3 / Top5 / Answer Correctness 四项百分比
- **Answer 饼图** — 正确 vs 错误占比
- **每题命中图** — 按问题展示 Top1 / Top3 / Answer 命中情况

### 导出

- **下载 CSV** — 评测结果的 CSV 文件（UTF-8 with BOM，Excel 友好）
- **下载 Markdown 报告** — 包含指标汇总、命中率表格、Top1 未命中列表、每题详情

## Judge 评分标准

Judge 支持两种 prompt 模板，根据样本是否带 reference_answer 自动选择：

| 字段 | 类型 | 说明 |
|------|------|------|
| `retrieval_top1_hit` | 0/1 | Top1 检索结果是否包含正确答案 |
| `retrieval_top3_hit` | 0/1 | Top3 检索结果中是否包含正确答案 |
| `retrieval_top5_hit` | 0/1 | Top5 检索结果中是否包含正确答案 |
| `answer_correct` | 0/1 | 最终回答是否正确（严格模式：与参考答案对比；合理性模式：基于检索内容判断） |
| `reason` | string | 评分理由（100 字以内） |

> **注意**：如果每题实际只召回 3 条检索结果，则 Top5 指标仅供参考，严格来说需要把 Dify 检索 topK 调到 5 后重新测试。

## 输出格式

### 样本 JSONL（`data/processed/langfuse_samples.jsonl`）

```json
{
  "trace_id": "abc123",
  "question": "P2P借贷是什么？",
  "retrieval_query": "P2P借贷",
  "retrieval_results": [
    {
      "position": 1,
      "score": 0.95,
      "document_name": "P2P借贷简介.md",
      "title": "P2P借贷简介",
      "content": "..."
    }
  ],
  "final_answer": "P2P借贷是指...",
  "reference_answer": "P2P借贷是点对点借贷...",
  "source_excerpt": "P2P借贷（Peer-to-Peer Lending）...",
  "difficulty": "基础",
  "topic": "P2P借贷"
}
```

### 评测结果 JSONL（`data/judged/eval_results.jsonl`）

```json
{
  "trace_id": "abc123",
  "question": "P2P借贷是什么？",
  "has_reference": true,
  "retrieval_top1_hit": 1,
  "retrieval_top3_hit": 1,
  "retrieval_top5_hit": 1,
  "answer_correct": 1,
  "reason": "Top1 检索结果包含相关内容，回答与参考答案一致"
}
```
