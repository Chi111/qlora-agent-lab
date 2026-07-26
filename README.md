# QLoRA Agent Lab

一个可以在 Windows + NVIDIA 单卡上完成的周日课程项目：

1. 使用 QLoRA 微调 Qwen 文本模型；
2. 使用 PyTorch、Transformers 和 FastAPI 封装 OpenAI 风格推理接口；
3. 使用 LangChain 或 Mastra Agent 接入本地模型；
4. Agent 通过 Tools 查询 Mock 订单后端、创建客服工单；
5. 使用 BGE-M3 + Qdrant + BGE Reranker 检索维修、产品、配送与售后知识；
6. 提供由 `@mastra/client-js` 驱动的 React 客服页面、流式回答、来源分数和工具审批；
7. 通过人工审核工作流，把脱敏后的历史维修经验保存为 Markdown 并增量写入向量库；
8. 使用独立评测集覆盖检索、Rerank、缺参数、超时、工具权限和提示词注入等边界。

## 运行架构

```text
React :5173 ── @mastra/client-js ──> Mastra Agent/API :4111
                                      ├─> QLoRA inference :8000（底座 + LoRA）
                                      ├─> Mock orders/tools :8001
                                      └─> RAG
                                           BGE-M3 :8003
                                              ↓ 1024 维
                                           Qdrant :6333（Top 12）
                                              ↓
                                           BGE Reranker :8003（Top 4）

LangChain Agent :8002 ──只读内部接口──> 同一套 Mastra RAG
```

服务是独立进程。Agent 不直接加载 GPU 模型，推理服务只加载一次模型，并把 GPU
生成请求串行化。不要为推理服务配置多个 Uvicorn worker，否则每个 worker 都会复制一份模型。
Embedding 与 Reranker 固定在 CPU 上；Rerank 失败时检索会关闭本次回答，不会退化为使用
未经重排的向量候选。

## 你的机器该选哪个配置

如果配置是：

- 24GB 系统内存
- RTX 5060 8GB 显存
- Ryzen 5 5600

请从 `configs/train_8gb.yaml` 开始。该配置默认训练 `Qwen/Qwen3-1.7B`、长度 512、
batch size 1。先确保闭环成功，再尝试更大的模型或更长上下文。

| GPU 显存 | 配置 | 默认模型 |
|---|---|---|
| 8GB | `configs/train_8gb.yaml` | `Qwen/Qwen3-1.7B` |
| 16GB | `configs/train_16gb.yaml` | `Qwen/Qwen3-4B` |
| 真实 24GB 显存 | `configs/train_24gb.yaml` | `Qwen/Qwen3-8B` |

Ollama 中的 `qwen3.5:9b` 是 GGUF 推理模型，不能直接作为该训练脚本的 QLoRA
基座。训练脚本会从 Hugging Face 下载单独的模型权重。项目提供
`scripts/serve_with_ollama.ps1`，可以在训练前先用已有 Ollama 模型测试 Agent 和 Tools。

> 当前训练配置针对纯文本 `AutoModelForCausalLM`。Qwen3.5-9B 是多模态架构，不要只把
> 8GB 配置里的模型名替换成它；8GB 显存也不适合在一天内训练 9B 多模态模型。

四份课程材料里的 Qwen3-8B 方案建立在 RTX 4090 的 **24GB 显存** 上。你的“24GB”
是系统内存，不会改变 RTX 5060 的显存上限，因此本项目保留同一套 QLoRA 方法，但把首轮
基座缩到 1.7B。训练出的仍是可插拔 LoRA adapter；等迁移到真实 24GB 显存机器时，再切换
`configs/train_24gb.yaml` 训练 Qwen3-8B。

## Windows 安装

### 1. 前置软件

- Windows 11；
- 较新的 NVIDIA 驱动；
- Python 3.11 x64；
- Git；
- Docker Desktop（本地 Qdrant）；
- Node.js 22.18 或更新版本（Mastra 与 React 前端）；
- 至少 30GB 可用磁盘；
- 建议把 Windows 分页文件设置为 24～32GB。

RTX 50 系显卡需要支持 Blackwell 的 PyTorch CUDA 构建。安装脚本默认使用 CUDA 12.8
wheel；如果 PyTorch 官方当前推荐了更新的索引，可通过参数覆盖。

### 2. 克隆与安装

在 PowerShell 中运行：

```powershell
git clone <本仓库地址>
cd qlora-agent-lab
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

如果 `py -3.11` 不可用：

```powershell
.\scripts\setup_windows.ps1 -Python "C:\Python311\python.exe"
```

如果 PyTorch 的 CUDA wheel 地址发生变化：

```powershell
.\scripts\setup_windows.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cu128"
```

脚本最后会运行环境预检，正常输出应包含：

```json
{
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 5060",
  "vram_gb": 7.XX,
  "bitsandbytes": "..."
}
```

若 `cuda_available` 为 `false`，先不要训练。检查驱动和 CUDA 版 PyTorch，而不是安装
完整 CUDA Toolkit。若原生 Windows 下 bitsandbytes 的 DLL 加载持续失败，建议改用
WSL2 + Ubuntu，项目代码和配置无需改变。

## 训练

先完全退出 Ollama、游戏和其他占显存程序，然后运行：

```powershell
.\scripts\train.ps1
```

也可以明确指定配置：

```powershell
.\scripts\train.ps1 -Config "configs/train_8gb.yaml"
```

训练完成后，LoRA adapter、tokenizer 和训练配置会保存到：

```text
artifacts/qlora-adapter/
```

`train.ps1` 会依次执行：

1. 校验 JSONL 格式、工具声明、重复样本和训练/评测集泄漏；
2. 检查 CUDA、bitsandbytes、GPU 型号、显存和所选模型档位；
3. 使用 NF4 + double quant 的 4-bit QLoRA 训练；
4. 只对 assistant 输出计算损失，并保存 adapter 与 tokenizer。

如果出现 CUDA OOM，按顺序调整：

1. 把 `max_length` 从 512 改为 384 或 256；
2. 把 `lora_r` 从 16 改为 8；
3. 保持 `batch_size: 1`；
4. 换成更小的 Qwen 模型；
5. 确认 Ollama 已停止。

不要先减少 `gradient_accumulation_steps` 来解决单步显存问题，它主要影响有效 batch，
并不能显著降低当前 micro-batch 的模型激活显存。

### 数据格式

`data/train.jsonl` 和 `data/eval.jsonl` 使用 TRL conversational/tool-calling 格式。
公共工具 JSON Schema 位于 `data/tools.json`，数据加载器会自动加入每个样本。

最小普通对话：

```json
{"messages":[
  {"role":"user","content":"帮我查下订单。"},
  {"role":"assistant","content":"请提供订单号。"}
]}
```

工具调用对话：

```json
{"messages":[
  {"role":"user","content":"查询 A100。"},
  {"role":"assistant","content":"","tool_calls":[
    {"type":"function","function":{
      "name":"get_order",
      "arguments":{"order_id":"A100"}
    }}
  ]},
  {"role":"tool","name":"get_order","content":"{\"ok\":true,\"data\":{\"status\":\"shipped\"}}"},
  {"role":"assistant","content":"订单 A100 已发货。"}
]}
```

当前示例包含 100 条训练对话和 22 条独立评测对话，覆盖订单查询、知识库检索、工单授权、
工具超时、隐私保护、提示注入、专业克制语气、情绪安抚、非客服问题拒答、混合意图和人工
转接等路径。它适合跑通课程并建立第一版客服行为基线，但仍不足以代表生产环境的真实分布。
上线前应持续加入经过脱敏和人工检查的真实领域数据，并保证评测数据不出现在训练集中。

## 启动完整 QLoRA 服务

训练完成后：

```powershell
.\scripts\serve_all.ps1
```

它会启动：

| 服务 | 地址 | 说明 |
|---|---|---|
| QLoRA inference | `http://127.0.0.1:8000` | OpenAI 风格模型接口 |
| Mock backend | `http://127.0.0.1:8001` | 订单和工单接口 |
| LangChain Agent | `http://127.0.0.1:8002` | 兼容原 Agent 接口，复用 Mastra RAG |
| BGE retrieval | `http://127.0.0.1:8003` | CPU Embedding 与 Rerank |
| Qdrant | `http://127.0.0.1:6333` | 1024 维维修知识向量库 |
| Mastra | `http://127.0.0.1:4111` | Agent、RAG、工作流与内部检索接口 |
| React web | `http://127.0.0.1:5173` | `@mastra/client-js` 客服页面 |

`serve_all.ps1` 现在会转到完整的 `serve_mastra.ps1`，因此推荐的客服页面是
`http://127.0.0.1:5173/`。原 LangChain API 仍位于 8002，并通过 Mastra 内部只读接口
使用同一套强制 Rerank 知识库。会话记录默认写入
`artifacts/conversations.sqlite3`。

脚本目前以独立 Windows 进程启动服务。完成测试后，可在任务管理器中结束对应 Python
进程。不要在模型服务上开启 `--reload` 或增加 `--workers`。

运行冒烟测试：

```powershell
.\scripts\smoke_test.ps1
```

也可以手动调用 Agent：

```powershell
$body = @{
  messages = @(
    @{
      role = "user"
      content = "帮我查询订单 A101。如果延迟了，先问我是否创建工单。"
    }
  )
  debug = $true
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/agent/invoke" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

## 使用 Mastra 接入训练后的模型

Mastra 版本位于 `mastra/`，使用 TypeScript/Node.js 编排 Agent，但模型推理仍由 Python
服务完成。它已经接入：

- `get_order`：实时查询订单，只把订单号、状态、商品和更新时间交给模型；
- `search_knowledge`：BGE-M3 向量化、Qdrant Top 12 召回、BGE Reranker 强制重排后取 Top 4；
- `create_ticket`：创建工单，带稳定幂等键，并强制要求 Mastra 人工批准；
- SQLite 多轮记忆；
- 中文客服范围、专业克制语气、非客服问题拒答和知识库提示注入边界。

### 1. 安装 Node.js 和 Mastra 依赖

安装 **Node.js 22.18 或更高版本（推荐当前 LTS）**。在项目目录运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_mastra.ps1
```

脚本会通过 `npx` 运行固定版本的 pnpm，并使用锁文件安装依赖，不受系统全局 pnpm 版本
影响。首次安装需要联网。无需为 Mastra 重新下载模型，也不需要再训练一次；它直接调用
`http://127.0.0.1:8000/v1` 上已经加载 LoRA adapter 的推理服务。

项目只批准 `esbuild` 执行依赖安装脚本，这是 Mastra 打包所必需的本地二进制；其他依赖
仍保持 pnpm 的默认禁止策略。如果曾看到 `ERR_PNPM_IGNORED_BUILDS`，同步最新代码后直接
重新运行 `setup_mastra.ps1`，无需执行交互式的 `pnpm approve-builds`。

### 2. 一键启动

训练完成并确认 `artifacts\qlora-adapter\adapter_config.json` 存在后运行：

```powershell
.\scripts\serve_mastra.ps1
```

脚本会启动 Qdrant（6333）、BGE 检索服务（8003）、Mock 后端（8001）、QLoRA 推理服务
（8000）、原 LangChain Agent（8002）、Mastra（4111）和 React 前端（5173），然后重建
并原子切换知识库别名。首次运行会下载 BGE 模型，时间取决于网络和 CPU。客服页面打开：

```text
http://127.0.0.1:5173
```

Mastra Studio/API 位于 `http://127.0.0.1:4111`。首次建议依次测试：

1. `帮我查询订单 A100。`
2. `七天内可以退货吗？`
3. `帮我写一段 Python 快速排序。`（应礼貌拒绝）
4. `订单 A101 延迟了，帮我创建一个高优先级工单。`

第 4 条会先出现工具审批。核对订单号、原因和优先级后再点击批准；拒绝审批不会写入工单。
可在另一个 PowerShell 窗口运行接口冒烟测试：

```powershell
.\scripts\smoke_test_mastra.ps1
.\scripts\evaluate_rag.ps1 -ApiKey "change-me-local-rag-key"
```

`evaluate_rag.ps1` 会让 30 条正例和 6 条无答案反例真实经过
BGE-M3 → Qdrant → BGE Reranker，输出 Hit@4、Top-1 和无答案准确率；加 `-Strict`
可按作业基线返回失败退出码。阈值需在本机实测后再调整，不要只凭单元测试宣称检索效果。

如果只修改了 Mastra 提示词或工具代码，不需要重训；如果要让 1.7B 模型本身更稳定地形成
新语气和拒答习惯，则同步新的 `data/train.jsonl` / `data/eval.jsonl` 后重新训练 adapter。

### 3. Mastra 配置

首次安装会从 `mastra\.env.example` 复制 `mastra\.env`。默认值已经适用于本项目：

```dotenv
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=local-qlora
MOCK_API_URL=http://127.0.0.1:8001
KNOWLEDGE_DIR=../data/knowledge
RAG_MODEL_SERVICE_URL=http://127.0.0.1:8003
RAG_MIN_RERANK_SCORE=0.2
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=repair_knowledge_current
MASTRA_INTERNAL_SEARCH_KEY=change-me-local-rag-key
```

SQLite 默认稳定写入 `mastra\mastra.db`；只有连接外部 libSQL 时才需要另设
`MASTRA_DB_URL`。

升级旧环境时，安装脚本会检查新增的 RAG 配置；若提示缺失，请把 `.env.example` 和
`mastra\.env.example` 中的新字段合并到已有文件。根目录的
`AGENT_RAG_INTERNAL_API_KEY` 必须与 Mastra 的 `MASTRA_INTERNAL_SEARCH_KEY` 完全一致，
否则原 LangChain Agent 会拒绝访问只读 RAG 接口。

这个演示没有用户登录和订单归属校验，只能监听本机地址测试。部署到局域网或公网前，必须
在 Mastra API 和订单后端前增加身份认证、授权、限流与审计。

### 4. 维修文档与向量入库

主知识文件：

- `data/knowledge/refrigerator-repair.md`
- `data/knowledge/television-repair.md`
- `data/knowledge/monitor-repair.md`

文档使用 YAML frontmatter 标注类型和版本。入库会按 Markdown 标题切片，生成稳定 ID，
写入完整来源 metadata。手动重建和检查：

```powershell
Set-Location mastra
pnpm rag:rebuild
pnpm rag:check
```

重建先写暂存集合，校验 1024 维和向量数量后再切换
`repair_knowledge_current` alias；失败时保留旧集合。

### 5. 过往维修经验工作流

`save-repair-experience` 工作流接收设备类型、故障现象、诊断、处理步骤、结果和安全提示。
它先生成 `data/knowledge/experience/drafts/*.md`，脱敏手机号、邮箱和证件号，并在人工审核
处暂停。只有批准后才移动到 `approved/` 并增量向量化；拒绝不会写入正式向量库。课程中的
向量数据库经验整理在
`data/knowledge/experience/approved/vector-database-experience.md`。

## 先用现有 Ollama 测试 Agent

确认 Ollama 中模型存在：

```powershell
ollama list
ollama run qwen3.5:9b
```

退出交互会话，但保持 Ollama 服务运行，然后执行：

```powershell
.\scripts\serve_with_ollama.ps1
```

该模式启动同一套 Qdrant、BGE、Rerank、Mastra、LangChain Agent 和 React 前端，只把
语言模型地址切换到 `http://127.0.0.1:11434/v1`。它用于先验证完整 RAG/Agent/Tools
闭环，不代表已经使用 QLoRA adapter。

## API

### 推理服务

```text
GET  /health
POST /v1/chat/completions
```

接口支持 `stream: false`，也支持 Mastra/AI SDK 所需的 OpenAI SSE 流式协议。当前流式
实现会先完成一次 GPU 生成，再把结果作为兼容的 SSE chunk 返回，并不是真正逐 token
输出。请求最多 32 条消息、默认最多生成 256 token。工具调用会从 Qwen 的
`<tool_call>...</tool_call>` 输出转换为 OpenAI `tool_calls`。服务会同时限制完整序列化
请求大小，并在分词后校验“输入 token + 输出 token”不超过模型上下文窗口。

### Mock 后端

```text
GET  /health
GET  /api/orders/A100
POST /api/tickets
```

内置订单：

- `A100`：已发货；
- `A101`：延迟；
- `A102`：处理中。

创建工单必须携带 `Idempotency-Key`。相同 key、相同参数的重试不会重复创建工单；相同
key 携带不同参数会返回 `409 IDEMPOTENCY_CONFLICT`，避免客户端误以为新参数已经生效。

### Agent

```text
GET  /health
POST /agent/invoke
POST /agent/chat
GET  /agent/sessions/{session_id}
POST /agent/sessions/{session_id}/handoff
```

只有 `debug: true` 才会返回脱敏后的消息和工具轨迹。生产环境不应把 debug 轨迹直接开放
给不受信任的客户端。

`/agent/chat` 是浏览器页面使用的多轮接口：服务端从 SQLite 恢复最近会话，再调用 Agent。
用户明确提出“转人工客服”时，由程序直接把会话切换为 `waiting_human`，不依赖模型是否
正确理解。该演示服务只监听 `127.0.0.1`；若改成公网服务，必须在会话和订单接口前增加
真实身份认证与订单归属校验。

### 本地 RAG

知识文件位于 `data/knowledge/*.md`。课程强调静态资料走 RAG、实时订单走 Tools，本项目
按这个边界实现：

- `search_knowledge`：产品说明、配送规则、退换货和退款政策；
- `get_order`：具体订单的实时状态；
- `create_ticket`：用户明确授权后的写操作。

`create_ticket` 不只依赖提示词：运行时会检查用户当前一轮是否明确要求创建工单，或是否
在客服询问后明确同意；否则工具返回 `WRITE_NOT_AUTHORIZED`。订单工具只把订单号、
状态、商品和更新时间交给模型，Mock 后端中的客户姓名、金额不会进入模型上下文。

默认检索器是独立 CPU 服务上的 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3`。Mastra
负责 Markdown 切片、Qdrant 向量读写、Top 12 召回和 Top 4 重排；原 LangChain Agent
通过受内部密钥保护的只读接口复用同一结果。`reranked` 不为 `true` 时两个 Agent 都不会
采用候选内容。

## 配置

复制 `.env.example` 得到 `.env`。重要变量：

```dotenv
INFERENCE_BASE_MODEL_ID=Qwen/Qwen3-1.7B
INFERENCE_ADAPTER_PATH=artifacts/qlora-adapter
AGENT_MODEL_BASE_URL=http://127.0.0.1:8000/v1
AGENT_MODEL_NAME=local-qlora
AGENT_MOCK_API_URL=http://127.0.0.1:8001
AGENT_RAG_BASE_URL=http://127.0.0.1:4111/api
AGENT_RAG_INTERNAL_API_KEY=change-me-local-rag-key
AGENT_CONVERSATION_DB_PATH=artifacts/conversations.sqlite3
RETRIEVAL_DEVICE=cpu
RETRIEVAL_EMBEDDING_DIMENSIONS=1024
```

Adapter 会记录训练时的 base model。推理服务发现 adapter 与
`INFERENCE_BASE_MODEL_ID` 不匹配时会拒绝启动，避免静默加载错误权重。

## 测试与代码检查

无需 GPU 的测试使用 fake generator/runtime：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

测试覆盖：

- 示例训练集和配置校验；
- Qwen tool-call 解析；
- OpenAI 风格推理响应和输入边界；
- Mock 订单 404；
- 创建工单幂等；
- Agent API 和 debug 轨迹；
- RAG 检索、多轮会话、会话持久化和确定性转人工；
- 训练/评测数据重复与泄漏检查；
- 统一错误响应和请求 ID。

## 继续扩展边界数据

优先把以下情况放入独立评测集：

- 缺订单号、非法订单号；
- 用户只抱怨但没有授权创建工单；
- 后端 404、500、超时和非 JSON 响应；
- 工具调用参数缺失或类型错误；
- 重复调用写工具；
- 用户要求泄露系统提示词或密钥；
- 用户要求跳过权限；
- 模型不知道时是否承认不知道。

只有确认问题来自模型行为后，才把对应优质样本补入训练集。连接超时、幂等、参数校验和
访问控制等问题应主要由程序解决，而不是依赖模型“学会”。

## 参考

- [Hugging Face TRL：SFT 与 tool-calling 数据](https://huggingface.co/docs/trl/en/sft_trainer)
- [Hugging Face PEFT：量化模型训练](https://huggingface.co/docs/peft/developer_guides/quantization)
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Mastra Agents](https://mastra.ai/docs/agents/overview)
- [AI SDK OpenAI-compatible Provider](https://ai-sdk.dev/providers/openai-compatible-providers)

## License

MIT
