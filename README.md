# Norma-CR
基于自定义代码规范驱动的自动化 Code Review Agent。支持多域（domain）标签审查、规则注入、并行处理与速率控制。

## 核心能力
- **标签化审查**：每个标签（STYLE/ERROR/API/CONC/PERF/SEC/TEST/CONFIG）都有专属 ReAct agent + 工具集，审查结果汇总为结构化 `FileCRResult`。
- **规则注入**：`coding-standards/registry.yaml` 中的规则按语言+domain 自动注入到 prompt，且可通过工具读取 Markdown 规则文档。
- **并行与限速**：使用 asyncio 一个 diff 内的文件审查并行进行；支持通过环境变量配置 QPS 限制。
- **可配置域/黑名单**：通过 profile YAML 为不同仓库指定 domains、文件黑名单、basename 黑名单，提供默认兜底配置。
- **报告生成**：LangGraph 末端节点生成 Markdown 报告，并写入 `code_review_report.md`。

## 快速开始
1) 安装依赖
```bash
uv sync
cp .env.example .env  # 填写 BASE_URL / API_KEY / MODEL_NAME 等
```

2) 准备 profile（示例见 `profiles/default.yaml`）
```yaml
version: 1
repos:
  - name: sample
    priority: 10
    match_paths: ["^/abs/path/to/your/repo(/|$)"]
    domains: ["SEC", "PERF"]
    skip_regex: ["^docs/generated/.*"]
    skip_basenames: ["README.md"]
default:
  domains: ["STYLE", "ERROR", "CONFIG"]
```

3) 运行
```bash
python agent.py \
  --repo /path/to/repo \
  --profile profiles/default.yaml \
  --env-file .env
```
参数优先级：命令行 > 环境变量 > `.env`。`CR_MAX_QPS` 可选，用于限速。

输出：终端概览 + 生成 `code_review_report.md`（默认写到仓库根目录，或通过 `CR_REPORT_DIR` 覆盖）。

## 目录速览
- `agent.py`：主入口，组装 LangGraph 流程（获取 commit diff -> 文件审查 -> 报告生成）。
- `cr_agent/file_review.py`：单文件审查引擎（标签打标、标签 agent 调用、结果合并）。
- `cr_agent/agents/`：Agent 抽象与 React 子类。
- `cr_agent/rules/`：规则加载、聚合与缓存；`RuleMeta.deprecated` 用于排除废弃规则。
- `coding-standards/`：规则注册表与规则文档。
- `docs/`：产品、规则、使用说明。

## 审查流程
1. **获取 diff**：`get_last_commit_diff` 读取最新提交的结构化 diff。
2. **文件打标**：LLM 按 diff 内容生成标签列表（空则兜底 STYLE）。
3. **标签审查**：为每个标签实例化 React agent，注入对应语言+domain 的规则（跳过 `deprecated:true`），必要时可用 `code_standard_doc(rule_id)` 读取 Markdown 文档。
4. **结果合并**：按文件聚合标签结果，生成 `FileCRResult`。
5. **报告生成**：LangGraph `render_report` 节点生成 Markdown（三部分：概述 / 带 rule_id 问题 / 无 rule_id 问题），`write_markdown_report` 将其落盘。


## 参考文档
- [产品设计](docs/product.md)
- [规则体系与编写](docs/rules.md)
- [使用说明](docs/usage.md)

