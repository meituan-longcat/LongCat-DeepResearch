![LongCat-DeepResearch](assets/longcat_logo.png)

<p align="center"><sub><strong>全局规划，独立研究，连贯综合。</strong></sub></p>

<p align="center"><a href="https://meituan-longcat.github.io/LongCat-DeepResearch/"><img alt="项目 Blog" src="https://img.shields.io/badge/Blog-Project_Page-2cb85b?style=for-the-badge"></a>&nbsp;<a href="https://github.com/meituan-longcat/LongCat-DeepResearch/blob/main/technical_report/LongCat-DeepResearch.pdf"><img alt="论文 PDF" src="https://img.shields.io/badge/Paper-PDF-d9534f?style=for-the-badge"></a>&nbsp;<a href="https://github.com/meituan-longcat/LongCat-DeepResearch"><img alt="GitHub 仓库" src="https://img.shields.io/badge/GitHub-Repository-24292f?style=for-the-badge&amp;logo=github"></a>&nbsp;<a href="https://github.com/meituan-longcat/LongCat-DeepResearch/blob/main/LICENSE"><img alt="MIT 许可证" src="https://img.shields.io/badge/License-MIT-5b6ea6?style=for-the-badge"></a></p>

<p align="center"><a href="https://github.com/meituan-longcat/LongCat-DeepResearch/blob/main/README.md">English</a> | <strong>简体中文</strong></p>

## **项目概览**

本仓库开源 LongCat-DeepResearch 的研究 Harness，用于执行开放式深度研究任务。它通过可执行的 ResearchSpec 记录逐步显现的研究需求，支持独立的分节调研与写作，并对组装后的报告进行有针对性的修订，减少反复重写整篇报告。用户按 Backend 协议实现 `llm`、`web_search` 和 `web_fetch` 三个方法，即可接入自己的服务运行深度研究任务。结合我们基于 LongCat-2.0、增强了深度研究能力的模型后，完整系统在 DeepResearchBench、DeepResearchBench II 和 ResearchRubrics 上均超过了 ChatGPT、Claude 和 Gemini 的 Deep Research 功能。

<span style="display:inline-block;width:660px;max-width:100%">![LongCat-DeepResearch 评测概览](assets/benchmark_overview.png)</span>

## **Research Harness**

![LongCat-DeepResearch Harness](assets/harness_overview.png)

Harness 包含三个阶段：

1. <strong>探索与规划。</strong>多个 Planner 调研问题和早期来源；Judge 合并方案，Critic 查找缺口，Reviser 生成可执行的 ResearchSpec。
2. <strong>分节研究。</strong>独立 Researcher 在分离的上下文中调研并撰写指定章节，同时保留完整任务书。
3. <strong>组装与编辑。</strong>完整章节直接组装；Global Editor 识别内容归属和一致性问题，Local Editor 执行章节级修订。

## **研究数据构建**

![基于证据的研究数据构建](assets/data_construction.png)

这套基于证据的研究数据构建流程从独立的 CC-BY 综述文章或冻结的多来源材料构造研究问题与任务级 Rubric，检查证据支持、可搜索性、泄漏和任务质量，并对通过验收的问题运行 Harness，收集和过滤分阶段轨迹。

本仓库仅发布研究 Harness，不包含研究数据构建流程实现、生成数据集、隐藏 Rubric、训练轨迹或训练代码。

## **快速开始**

Harness 本身只使用 Python 标准库，支持 Python 3.10 及以上版本。首先以可编辑模式安装：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
~~~

要运行它，需要提供一个实现以下三个方法的 Python 对象：

- <code>llm</code>：把 OpenAI Chat Completions/function-calling 格式的请求发送给你的模型，并把服务商响应作为字典返回。
- <code>web_search</code>：接收一条查询，返回统一格式的搜索结果。
- <code>web_fetch</code>：接收一个或多个网址，按照请求顺序返回统一格式的网页正文。

本仓库不内置托管 Backend，因为模型、搜索和网页抓取服务通常需要账号专属凭据。请在仓库根目录创建 <code>local_backend.py</code>。该文件已经被 Git 忽略，因此凭据和服务商专属实现只保留在本地：

~~~python
class Backend:
    def llm(self, request: dict) -> dict:
        # 输入：{"model", "messages", "stream", "max_tokens", ...}
        # 输出：{"choices": [{"message": {...}, "finish_reason": "..."}]}
        raise NotImplementedError

    def web_search(self, request: dict) -> dict:
        # 输入：{"query": str, "top_k": int}
        # 输出：{"results": [{"title": str, "url": str,
        #                      "snippet": str, "published_at": str | None}]}
        raise NotImplementedError

    def web_fetch(self, request: dict) -> dict:
        # 输入：{"urls": [str, ...]}
        # 输出：{"pages": [{"url": str, "title": str,
        #                   "text": str, "error": str | None}, ...]}
        raise NotImplementedError


def create_backend():
    return Backend()
~~~

设置 Backend 模块和模型名称，然后先验证三个外部服务。Smoke 命令只会分别发起一次小型模型请求、搜索请求和网页抓取请求，不会生成研究报告：

~~~bash
export DR_BACKEND_MODULE=local_backend
export DR_MODEL=your-model-name

scripts/run_smoke.sh
~~~

完整研究任务使用独立的 Standard 启动脚本：

~~~bash
export DR_BACKEND_MODULE=local_backend
export DR_MODEL=your-model-name

printf '%s\n' '需要研究的问题' > question.txt
scripts/run_standard.sh question.txt runs
~~~

最终报告位于 <code>runs/&lt;question_hash&gt;/report_final.md</code>。规划、分节调研和编辑等中间产物也会保存在同一个运行目录中，便于检查。

作为 Python 库使用时，也可以直接注入 Backend：

~~~python
from longcat_deepresearch import LongCatDeepResearch

result = LongCatDeepResearch(backend=my_backend).run(
    "需要研究的问题",
    run_root="runs",
)
print(result.final_path)
~~~

## **Backend 协议**

Backend 是 Harness 与外部服务之间的传输边界：Harness 决定何时调用模型和工具，Backend 负责鉴权以及如何访问具体服务。Harness 会在 <code>llm</code> 请求中把 <code>web_search</code> 和 <code>web_fetch</code> 声明为 OpenAI function-calling 格式的工具；如果模型返回工具调用，Harness 会执行对应的 Backend 方法，并在下一轮把工具结果发回模型。

公开接口定义在 [<code>longcat_deepresearch/backend.py</code>](longcat_deepresearch/backend.py)。每个方法接收一个 Python 字典，并返回一个 Python 字典：

| 方法 | Harness 提供的请求 | Backend 必须返回的响应 |
|---|---|---|
| <code>llm</code> | <code>model</code>、<code>messages</code>、<code>stream=false</code>、<code>max_tokens</code>，以及可选的 <code>tools</code>/<code>tool_choice</code> | 包含 <code>choices[0].message</code> 的 OpenAI Chat Completions 兼容对象 |
| <code>web_search</code> | <code>{"query": "...", "top_k": 8}</code>；<code>top_k</code> 范围为 1–10 | <code>{"results": [...]}</code>，每项包含统一格式的标题、网址、摘要和可选发布日期 |
| <code>web_fetch</code> | <code>{"urls": ["https://..."]}</code>，包含 1–10 个 HTTP(S) 网址 | <code>{"pages": [...]}</code>，每个请求网址对应一项，并保持请求顺序 |

模型直接给出文本答案时，<code>llm</code> 返回：

~~~json
{
  "choices": [
    {
      "message": {"role": "assistant", "content": "回答或报告正文。"},
      "finish_reason": "stop"
    }
  ]
}
~~~

模型要求调用工具时，<code>message.content</code> 可以是 <code>null</code>。其中 <code>function.arguments</code> 必须是 JSON 编码后的字符串，不能直接放嵌套对象：

~~~json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "{\"query\":\"example\",\"top_k\":5}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
~~~

统一格式的搜索响应如下：

~~~json
{
  "results": [
    {
      "title": "来源标题",
      "url": "https://example.org/source",
      "snippet": "简短的搜索结果摘要。",
      "published_at": "2026-01-01"
    }
  ]
}
~~~

统一格式的网页抓取响应如下：

~~~json
{
  "pages": [
    {
      "url": "https://example.org/source",
      "title": "来源标题",
      "text": "可读的网页正文。",
      "error": null
    }
  ]
}
~~~

字段级规则和批量网页抓取示例见[完整 Backend 接口说明](docs/BACKEND_INTERFACE.md)。Backend 负责鉴权、网络请求、重定向、超时、重试、限流、服务商专属字段、响应大小限制、线程安全，以及私网和链路本地地址阻断。

## **仓库结构**

~~~text
.
├── assets/                 # README 图片与视觉资源
├── docs/                   # 详细 Backend 协议说明
├── longcat_deepresearch/   # 可安装的 Python 包和命令行入口
├── scripts/                # 命令行入口和发布检查
├── technical_report/       # 最新技术报告 PDF
├── tests/                  # 离线接口测试和端到端测试
└── pyproject.toml          # 包元数据和开发工具配置
~~~

## **测试**

所有测试均使用假 Backend，不会发起外部请求。

~~~bash
python -m pip install -e ".[dev]"
python3 -m unittest discover -s tests -v
ruff check . --exclude .venv
python3 scripts/open_source_preflight.py
~~~

## **限制与负责任使用**

生成报告可能包含缺乏依据的结论、错误引用、不安全建议，或编辑阶段造成的信息遗漏。使用者必须人工审核输出，遵守网站条款及 robots 规则，保护个人信息，并确保接入的模型、数据集和服务已经获得授权。

Backend 必须将检索内容视为不可信输入，阻止访问私网和链路本地地址，保护凭据，并避免记录鉴权 Header 或私有 Prompt。

请勿在公开 Issue 中披露漏洞。应使用仓库提供的私密安全报告渠道，并附上受影响版本、复现步骤和预期影响。

## **引用**

如果本项目对你的工作有帮助，请引用：

```bibtex
@techreport{longcatdeepresearch2026,
  title  = {LongCat-DeepResearch Technical Report},
  author = {
    Meituan LongCat Team and He Zhu and Yue Xu and
    Xunliang Cai and Yan Chen and Fan Yang and
    Lingchuan Liu and others
  },
  year   = {2026}
}
```

## **许可证**

Copyright 2026 LongCat。

本项目基于 [MIT License](LICENSE) 发布。
