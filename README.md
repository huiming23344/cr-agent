# Norma-CR
基于既有的私有代码规范规则驱动的 Code Review Agent

主要特性：
1. 并行/限速审查：对一个 diff 中所有文件并行发起 tag+子 agent，
2. LLM client 级别的可配置并发度限制。
3. 标签驱动：每个 tag 对应拥有独立 prompt + 工具集 的agent，结果会汇总成结构化的 `FileCRResult`
4. 不同代码库可对应不同 domain/黑名单：通过 profile YAML 指定 `match_paths`、`domains`、`skip_regex`、`skip_basenames`，并提供 default 兜底。
5. 不同使用场景可以针对一个仓库同过 profile YAML 做多种配置。




