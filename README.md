# Langfuse RAG 评测工具

RAG 检索 + 回答质量评测工具。从知识库生成题目，通过 Dify 批量提问，解析为结构化样本后用 LLM Judge 自动评分。运行看板按配置方案汇总累计指标、运行历史和单次运行详情。

## 功能概览

- **题目生成** — 上传知识库文件，自动按章节切分后调用 LLM 生成带参考答案的评测题集
- **批量提问** — 选择题集和 RAG 配置方案，通过 Dify Workflow API 批量提问，收集回答与检索结果
- **样本准备** — 解析 Dify / Langfuse 记录为结构化样本，回填参考答案和运行元数据
- **Judge 评测** — 按评测轨道自动评分：检索评测关注 Top1/3/5 命中，问答评测关注正确性/合理性
- **运行看板** — 按配置方案查看累计结果、运行历史和单次运行详情，支持安全编辑配置描述
- **评测优化** — 规则预筛选 + 内容级去重 + 分层 prompt 截断，减少 LLM 调用次数
- **报告导出** — 一键下载 CSV 或 Markdown 评测报告

## 四步工作流 + 运行看板

```
题目生成 → 批量提问 → 样本准备 → Judge 评测
                                        ↓
                                   运行看板（累计指标 + 运行历史 + 单次详情）
```

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1. 题目生成 | `question_generator.py` | 上传知识库文件，调用 LLM 生成题集，输出 question / reference_answer / source_excerpt / question_set_id |
| 2. 批量提问 | `batch_query.py` | 选择题集 + 配置方案，通过 Dify API 批量提问，产出 raw 文件（含 run_id / config_id） |
| 3. 样本准备 | `parser.py` | 解析 raw 文件为 processed samples（使用真实 Langfuse trace_id），回填参考答案和元数据 |
| 4. Judge 评测 | `judge.py` | 按评测轨道（retrieval / strict_qa / grounded_qa）调用 LLM 评分 |
| 5. 运行看板 | `experiment.py` + `app.py` | 按配置方案聚合累计指标（加权汇总），查看运行历史和单次详情 |

### 数据关联链

```
run_id → processed sample（真实 Langfuse trace_id）→ Judge result
```

- `batch_qa_*` 是批量提问生成的文件标识，**不是** Langfuse trace_id
- Judge 结果通过 processed sample 的 trace_id 关联，不通过 batch_qa_* 关联
- 历史 Judge 结果没有 run_id 时，通过 trace_id fallback 关联
- 运行看板累计指标按有效 Judge 样本数加权汇总，不是各 run 百分比的简单平均

## 评测轨道

| 轨道 | 触发条件 | 核心指标 |
|------|---------|---------|
| **retrieval（检索评测）** | question_mode=retrieval 且有金标准证据 | Top1 / Top3 / Top5 Hit |
| **strict_qa（严格问答）** | question_mode=qa 且有 reference_answer | Answer Correctness |
| **grounded_qa（合理性问答）** | question_mode=qa 且无 reference_answer | Answer Groundedness |
| **not_evaluable** | 检索评测题但缺少金标准证据 | 不纳入计算 |

## 配置方案字段

### 必填字段

| 字段 | 说明 |
|------|------|
| `config_name` | 配置名称 |
| `knowledge_base_version` | 知识库版本 / 文档版本 |
| `workflow_version` | 工作流名称或版本 |

### 可选字段

| 字段 | 说明 |
|------|------|
| `source_description` | 文档 / 数据来源说明 |
| `chunk_strategy` | 分块策略 |
| `embedding_model` | Embedding 模型 |
| `retrieval_mode` | 检索模式（如 hybrid / semantic） |
| `retrieval_config` | 检索配置说明 |
| `top_k` | Top K（整数） |
| `rerank_model` | Rerank 模型 |
| `changed_variable` | 本次改动 |
| `notes` | 备注 |

### 只读核心字段

`config_id`、`created_at` — 不可在 UI 编辑。

## 项目结构

```
Langfuse_test/
├── app.py                    # Streamlit 主界面（5 个 Tab）
├── experiment.py             # 运行看板模块（配置方案 + 运行记录 + 字段 schema）
├── judge.py                  # Judge 评测模块（prompt 构建、API 调用、指标计算）
├── parser.py                 # 样本准备模块（解析 + 回填参考答案和元数据）
├── question_generator.py     # 题目生成模块
├── batch_query.py            # 批量提问模块
├── fetch_traces.py           # Langfuse API Trace 拉取模块
├── main.py                   # CLI 入口（仅解析，不含 Judge）
├── prompts/
│   ├── judge_prompt.txt           # Judge Prompt（合理性问答模板）
│   ├── judge_prompt_with_ref.txt  # Judge Prompt（严格问答模板）
│   ├── judge_prompt_retrieval.txt # Judge Prompt（检索评测模板）
│   └── qgen_prompt.txt            # 题目生成 Prompt
├── data/
│   ├── raw/                  # 批量提问推送的 raw 文件（batch_qa_*.jsonl）
│   ├── processed/            # 解析后的结构化样本
│   │   ├── langfuse_samples.jsonl   # Judge 评测的直接输入
│   │   └── langfuse_summary.json
│   ├── judged/               # Judge 评测结果
│   │   ├── eval_results.jsonl       # 最新结果（持续积累）
│   │   └── eval_results_<时间戳>.jsonl  # 历史快照
│   ├── questions/            # 生成的题集文件（含 question_set_id）
│   ├── batch/                # 批量提问完整结果（含成功/失败状态）
│   ├── config_profiles/      # 配置方案 JSON 文件
│   └── experiments/          # 运行记录（每个 run 一个目录，含 manifest.json）
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
2. 选择出题模式（检索评测 / 全流程问答评测）
3. 设置生成数量、难度偏好、生成策略
4. 点击「生成题目」
5. 题集保存到 `data/questions/`，含 question、reference_answer、source_excerpt、question_set_id 等

### 3. 批量提问

1. 在「批量提问」tab 选择问题来源（已生成题目 / 手动输入 / 文件加载 / 历史题集）
2. 创建或选择 RAG 配置方案（记录知识库版本、工作流版本等描述性参数）
3. 配置 Dify API Key 和地址
4. 点击「开始提问」
5. 成功结果自动推送到 `data/raw/`，同时创建运行记录（run_id + config_snapshot）

### 4. 样本准备

1. 在「样本准备」tab 选择 raw 文件（或上传 Langfuse JSONL / 从 API 拉取）
2. 点击「开始解析」
3. 解析结果保存到 `data/processed/langfuse_samples.jsonl`
4. 自动从题目库回填 reference_answer，从 user_id 回填 run_id 等元数据
5. 解析后显示回填统计

### 5. Judge 评测

1. 在「Judge 评测」tab 配置 API（Key、Base URL、Model）
2. 选择评测范围和模式（跳过已有结果 / 强制重评）
3. 可点击「预览优化策略」查看实际 LLM 调用次数
4. 点击「运行 Judge 评测」
5. 查看指标、可视化图表、评测详情（按评测轨道分组展示）

### 6. 运行看板

1. 在「运行看板」tab 选择配置方案
2. 查看配置方案总览（累计 Judge 指标，按样本数加权汇总）
3. 展开单次运行查看该 run 的图表和逐题明细
4. 可编辑配置方案的描述性字段，或修正某次运行的配置快照
5. 查看运行历史趋势图

### 7. CLI 模式（仅解析）

```bash
python main.py <input.jsonl> [--output PATH] [--summary PATH]
```

## 历史数据兼容

- 旧格式 Judge 结果（无 run_id）：运行看板通过 processed trace_id fallback 正确关联
- 旧格式配置（无新字段）：显示"未记录"，可随时补充
- 旧格式 processed 样本（无 question_set_id）：从 user_id 解析 run_id 后从 manifest 回填
- 数据迁移工具：在运行看板的「数据迁移工具」折叠区可执行批量回填

## 常见问题

### Raw / Judge 显示为 0

1. 确认 batch 文件已推送到 `data/raw/`
2. 确认样本准备已解析 raw 文件（`data/processed/langfuse_samples.jsonl` 存在且非空）
3. 确认 Judge 使用的 trace_id 与 processed sample 的 trace_id 一致（不是 batch_qa_*）
4. 运行看板通过 `run_id → processed trace_id → judged trace_id` 链路关联，确认链路完整

### 配置方案字段不一致

批量提问、运行看板编辑、运行快照修正使用同一套字段 schema（定义在 `experiment.py` 的 `CONFIG_FIELD_SCHEMA`）。
编辑配置方案不影响历史运行的 config_snapshot；修正运行快照不影响配置方案。

## 测试

```bash
python test_experiment.py            # 配置方案和运行记录基础测试
python test_experiment_dashboard.py  # 运行看板测试
python test_experiment_e2e.py        # 端到端测试（batch → raw → processed → judged）
python test_experiment_e2e_v2.py     # 端到端测试 v2（batch trace_id 与 Langfuse trace_id 不同）
python test_experiment_viz.py        # 可视化增强测试（图表、筛选、跨 run 隔离）
python test_evaluation_tracks.py     # 评测轨道分类和指标计算测试
python test_charts_and_details.py    # 图表 layout 和详情展示测试
python test_question_mode.py         # 出题模式测试
python test_question_set.py          # 题集管理测试
python test_result_status.py         # 结果状态显示测试
python test_config_overview.py       # 配置方案总览测试（加权汇总、去重、隔离）
python test_history_selector.py      # 历史题集选择器测试（标签唯一性、选择绑定）
python test_safe_edit.py             # 安全编辑测试（核心字段保护、快照审计）
python test_chart_layout.py          # 图表 layout 单元测试（margin、title、height）
python test_unified_config.py        # 统一配置 schema 测试（跨 Tab 一致性）
```

## 输出格式

### 样本 JSONL（`data/processed/langfuse_samples.jsonl`）

```json
{
  "trace_id": "真实 Langfuse UUID",
  "question": "P2P借贷是什么？",
  "retrieval_query": "P2P借贷",
  "retrieval_results": [
    {
      "position": 1,
      "score": 0.95,
      "document_name": "P2P借贷简介.md",
      "content": "..."
    }
  ],
  "final_answer": "P2P借贷是指...",
  "reference_answer": "P2P借贷是点对点借贷...",
  "source_excerpt": "P2P借贷（Peer-to-Peer Lending）...",
  "question_mode": "retrieval",
  "question_set_id": "qs_20260713_164111_...",
  "run_id": "run_20260713_170321_...",
  "config_id": "cfg_20260713_170321_..."
}
```

### 评测结果 JSONL（`data/judged/eval_results.jsonl`）

```json
{
  "trace_id": "真实 Langfuse UUID",
  "question": "P2P借贷是什么？",
  "evaluation_track": "retrieval",
  "has_reference": true,
  "retrieval_top1_hit": 1,
  "retrieval_top3_hit": 1,
  "retrieval_top5_hit": 1,
  "answer_correct": 1,
  "reason": "Top1 检索结果包含相关内容，回答与参考答案一致",
  "run_id": "run_20260713_170321_..."
}
```
