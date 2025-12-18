# cr-agent
基于既有的私有代码规范，以错误检出率优先、规则驱动的 Code Review Agent，降低团队人工审查成本并提升规范一致性。


主要特性：
1. 并行/限速审查：对一个 diff 中所有文件并行发起 tag+子 agent，
2. LLM QPS限速可配置。
3. 标签驱动：每个 tag 对应拥有独立 prompt + 工具集 的agent，结果会汇总成结构化的 `FileCRResult`
4. 不同代码库可对应不同 domain/黑名单：通过 profile YAML 指定 `match_paths`、`domains`、`skip_regex`、`skip_basenames`，并提供 default 兜底。
5. 不同使用场景可以针对一个仓库同过 profile YAML 做多种配置。


## 使用说明

- CLI 启动示例：
  ```bash
  python agent.py \
    --repo /path/to/repo \
    --profile profiles/default.yaml \
    --env-file .env
  ```
  参数/环境变量优先级：命令行 > shell中手动注册的环境变量 > 指定的`.env`文件。未指定 profile 时默认不过滤 domain 与文件（即标签全开）。

- Rate limit：可通过 `.env` 或环境变量配置 `CR_MAX_QPS`（正数/小数），用于限制 LLM 对每个文件标签/审查请求的 QPS；未填写则不做限速。

- profile YAML（如 `profiles/default.yaml`）结构：
  ```yaml
  version: 1
  repos:
    - name: foo-service
      priority: 10
      match_paths: ["^/Users/you/work/foo-service(/|$)"]
      domains: ["SEC", "PERF"]
      skip_regex: ["^docs/generated/.*", "\\.pb\\.go$"]
      skip_basenames: ["README.md"]
  default:
    domains: ["STYLE", "ERROR", "CONFIG"]
  ```
  `repos` 列表按 `priority`（数值越小越优先）排序，匹配 `repo_path` 后使用对应 domains/黑名单；若都不匹配则使用 `default` 中配置。

- 环境变量中只需配置 `CR_PROFILE_PATH`，而白名单/黑名单默认完全禁用，全部逻辑由 profile 控制。

## 部署
- 先clone仓库到本地，使用uv进行包管理
- 依赖安装：`uv sync`
- 复制```.env.example``` 文件，补全其中的内容

