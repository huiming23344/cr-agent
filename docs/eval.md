# 规则检出率评测套件（eval）

该模块用于批量评估特定规则（rule_id）的检出率/误报率。核心思路：构造带有期望结果的 diff 样本，用现有审查链路跑完整流程，解析输出的 NDJSON 结果并统计。

## 目录结构
- `eval/cases/<RULE_ID>/<case-name>/`
  - `base/`（可选）：基线文件内容，运行前会拷贝到临时 repo。
  - `patch.diff`：统一 diff（与 `git diff` 输出一致），在临时 repo 上应用并提交。
  - `expect.yaml`：预期结果。
- `profiles/eval.yaml`：评测用 profile，默认开启所有 domain（可按需调整）。
- `results/`：评测输出目录（默认 `results/eval.ndjson`）。

### expect.yaml 示例
```yaml
rule_id: GO-ERROR-002
domain: ERROR
language: go
expect_hit: true         # 该 case 是否应命中此规则
expect_count: 1          # 预期至少命中次数（可选）
allow_other_rules: false # 是否允许其他 rule 同时命中（可选）
```

## 工作流程
1. 遍历 `eval/cases/**/expect.yaml`。
2. 为每个 case 创建临时 git 仓库：
   - 初始化 repo 并创建空的 baseline commit（避免无父提交）。
   - 拷贝 `base/`（如有）并提交。
   - 应用 `patch.diff`，提交为 “case”。
3. 设置 `CR_EVAL_MODE=1` 调用 `agent.py`，复用完整的审查链路。`CR_EVAL_MODE` 会额外生成 NDJSON 报告（每条 issue 一行 JSON）。
4. 读取 NDJSON，按 `rule_id` 判断命中，写出汇总结果。

## 运行方式
```bash
python tools/eval_rules.py \
  --cases eval/cases \
  --profile profiles/eval.yaml \
  --env-file .env \
  --out results/eval.ndjson
```

- `--cases`：测试用例根目录。
- `--profile`：评测使用的 profile。
- `--env-file`：传入 API 配置等（同 `agent.py`）。
- `--out`：汇总结果输出（NDJSON，每行一个 case 结果）。

## NDJSON 输出（测试模式）
评测运行时 `CR_EVAL_MODE=1`，`agent.py` 会在生成 Markdown 报告的同时输出 NDJSON（文件名形如 `cr_report_<short_sha>.ndjson`），示例行：
```json
{"rule_id": "GO-ERROR-002", "file": "pkg/foo/service.go", "line_start": 12, "line_end": 18, "hit": true, "severity": "error", "message": "...", "path_line": "pkg/foo/service.go:12-18", "approved": false}
```
汇总文件 `results/eval.ndjson` 则记录每个 case 的命中情况：
```json
{"case": "missing_export_comment", "rule_id": "GO-STYLE-001", "expect_hit": true, "actual_hit": true, "expected_count": 1, "actual_count": 1, "allow_other_rules": true}
```

## 用例编写注意
- `patch.diff` 必须是标准 unified diff，包含 `diff --git`、`---/+++`、`@@` 块，确保 `git apply` 可用。
- 正例/负例都可添加：正例用于检出率，负例用于误报率。
- 如需上下文，放在 `base/`；无上下文可省略。

## 常见问题
- NDJSON 为空或 patch 被跳过：检查 `patch.diff` 是否有效（空文件或无 hunk 会被跳过）。
- 初始提交无父提交：脚本已创建 baseline 空提交，若手动运行需保证有父提交。
- 规则未命中：确认 `expect.yaml` 的 `rule_id` 是否启用，且 profile 中 domain 是否包含目标标签。
