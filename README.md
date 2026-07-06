# Langfuse Event Export Data Processor

将 Langfuse 导出的 JSONL 事件数据转换为按 trace 聚合的结构化样本记录，适用于 RAG 知识库聊天机器人工作流分析。

## 功能概述

- 读取 Langfuse 导出的 JSONL 文件（每行一条 observation/span）
- 按 `traceId` 对观测数据进行分组
- 提取每条 trace 的关键信息：用户问题、检索结果、LLM 模型/输入/输出、最终回答等
- 输出结构化的 JSONL 样本文件和摘要 JSON 文件

## 环境要求

- Python 3.13+
- 无第三方依赖，仅使用 Python 标准库（`argparse`、`json`、`collections`、`pathlib`）

## 使用方法

```bash
python main.py <input.jsonl> [--output PATH] [--summary PATH]
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `input` | 是 | Langfuse 导出的 `.jsonl` 文件路径 |
| `--output` | 否 | 输出 JSONL 文件路径，默认为 `<input>.samples.jsonl` |
| `--summary` | 否 | 摘要 JSON 文件路径，默认为 `<input>.summary.json` |

### 示例

```bash
# 使用默认输出路径
python main.py 1783322412151-lf-events-export-cmr8tec2s0007ad0c2klqym7l.jsonl

# 指定输出路径
python main.py data.jsonl --output result.jsonl --summary summary.json
```

## 输出格式

### 样本 JSONL（每行一条 trace 记录）

```json
{
  "trace_id": "abc123",
  "trace_name": "workflow_name",
  "session_id": "session_001",
  "user_id": "user_001",
  "workflow_run_id": "run_001",
  "question": "P2P借贷是什么？",
  "root_input": {},
  "root_output": {},
  "retrieval_query": "P2P借贷",
  "retrieval_results": [
    {
      "position": 1,
      "score": 0.95,
      "document_name": "金融知识库",
      "segment_id": "seg_001",
      "chunk_id": "chunk_001",
      "node_type": "knowledge-retrieval",
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

### 摘要 JSON

```json
{
  "input_file": "input.jsonl",
  "output_file": "input.samples.jsonl",
  "trace_count": 10,
  "bad_line_count": 0,
  "bad_lines": []
}
```

## 核心逻辑

| 字段 | 提取规则 |
|------|----------|
| `question` | 从 observation input 的 `sys.query` 或 `query` 字段提取 |
| `retrieval_results` | `node_type == "knowledge-retrieval"` 或名称包含"知识"的节点 |
| `llm_model` | `type == "GENERATION"` 或 `node_type == "llm"` 或 `name == "LLM"` 的节点 |
| `final_answer` | 优先从 `answer` 节点提取，其次从 LLM output 的 `text` 提取 |

## 项目结构

```
Langfuse_test/
├── main.py                          # 主程序（209 行）
├── *.jsonl                          # Langfuse 导出的原始数据文件
├── .venv/                           # Python 虚拟环境
└── README.md                        # 本文档
```
