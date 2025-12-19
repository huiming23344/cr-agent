# Coding Standards

在 Coding Standards 目录下集中管理代码规范，包括规范注册表、运行 profile 以及具体规则描述。结构如下：

- `registry.yaml`：记录当前支持的规则集合，以及各语言、Profile 的映射。
- `rules/<lang>/`：具体规则详情文档，按语言组织。

添加新规则时，建议同步更新 `registry.yaml` 并在相应语言目录下创建 Markdown 说明文件。

简要流程梳理：获取`registry.yaml`注册的所有rule，作为prompt，大模型自行决定是否调用tool获取md文档进行判断

## registry 字段说明
```yaml
    # id: 作为唯一标识，废弃了也不建议更换
  - id: GO-STYLE-001

    # 可选，废弃的字段不会添加进大模型传入
    deprecated: false
    
    title: "导出符号必须提供注释，保持可读性" 

    language: go 

    # severity: 可选，当前并没有使用，会作为模型传入的一部分，建议留空，此处占位为了后续拓展
    severity: info  

    # domains: 列表，标识在哪些 domain agent 中进行检查，其中的值必须是下面domain中的一种
    # "STYLE": "风格/可读性（命名、结构、注释、可维护性）",
    # "ERROR": "错误处理（边界、异常、返回值、降级、日志）",
    # "API": "接口设计（对外契约、兼容性、参数与返回、文档）",
    # "CONC": "并发（线程安全、竞态、锁、异步、资源释放）",
    # "PERF": "性能（复杂度、IO、缓存、分配、热点）",
    # "SEC": "安全（鉴权、输入校验、注入、敏感信息、权限）",
    # "TEST": "测试（覆盖率、用例质量、回归、稳定性）",
    # "CONFIG": "配置/依赖（配置项、环境变量、依赖版本、部署影响）",
    domains: [STYLE] 

    # path: 可选，文件路径，对于简单规则，可以不整理文档
    path: "coding-standards/rules/go/GO-STYLE-001.md" 

    # prompt_hint: 可选，但强烈建议添加，规范的详细说明
    prompt_hint: >  
      检查新增/修改的导出类型、函数、常量和接口是否带有 Go 风格注释（以名称开头）。
      确保注释解释用途、关键约束以及并发/性能注意事项，避免信息缺失。
```
id, title, severity, prompt_hint 四个字段会被直接作为输入传入到模型中

## 创建rule

prompt：
> ```yaml
>     # id: 作为唯一标识
>  - id: GO-STYLE-001
>   
>    title: "导出符号必须提供注释，保持可读性" 
>
>    language: go 
>
>    # severity: 可选
>    severity: info  
>
>    # domains: 列表，标识在哪些 domain agent 中进行检查，其中的值必须是下面domain中的一种
>    # "STYLE": "风格/可读性（命名、结构、注释、可维护性）",
>    # "ERROR": "错误处理（边界、异常、返回值、降级、日志）",
>    # "API": "接口设计（对外契约、兼容性、参数与返回、文档）",
>    # "CONC": "并发（线程安全、竞态、锁、异步、资源释放）",
>    # "PERF": "性能（复杂度、IO、缓存、分配、热点）",
>    # "SEC": "安全（鉴权、输入校验、注入、敏感信息、权限）",
>    # "TEST": "测试（覆盖率、用例质量、回归、稳定性）",
>    # "CONFIG": "配置/依赖（配置项、环境变量、依赖版本、部署影响）",
>    domains: [STYLE] 
>
>    # path: 可选，文件路径，对于简单规则，也可以不整理文档
>    path: "coding-standards/rules/go/GO-STYLE-001.md" 
>
>    # 规范的详细说明
>    prompt_hint: >  
>      检查新增/修改的导出类型、函数、常量和接口是否带有 Go 风格注释（以名称开头）。
>      确保注释解释用途、关键约束以及并发/性能注意事项，避免信息缺失。
>```
> 
>请你参照以上代码，给出在go语言中以下规范的yaml配置，不要保留任何注释：
> 
