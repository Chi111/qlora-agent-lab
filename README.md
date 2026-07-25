# QLoRA Agent Lab

一个可以在 Windows + NVIDIA 单卡上完成的周日课程项目：

1. 使用 QLoRA 微调 Qwen 文本模型；
2. 使用 PyTorch、Transformers 和 FastAPI 封装 OpenAI 风格推理接口；
3. 使用 LangChain Agent 接入本地模型；
4. Agent 通过 Tools 查询 Mock 订单后端、创建客服工单；
5. 使用本地知识库回答产品、配送与售后问题；
6. 提供浏览器客服页面、SQLite 多轮会话留存和确定性的转人工入口；
7. 使用独立评测集覆盖缺参数、404、超时、工具权限和提示词注入等边界。

## 运行架构

```text
                         ┌──────────────────────────┐
                         │ QLoRA inference :8000    │
浏览器 ──> Agent :8002 ──> │ /v1/chat/completions    │
              │           └──────────────────────────┘
              ├─────────> 本地 Markdown 知识库（CPU BM25 检索）
              ├─────────> SQLite 会话记录 / 人工接管状态
              └─────────> Mock backend :8001
                           ├─ GET  /api/orders/{id}
                           └─ POST /api/tickets
```

三个服务是独立进程。Agent 不直接加载 GPU 模型，推理服务只加载一次模型，并把 GPU
生成请求串行化。不要为推理服务配置多个 Uvicorn worker，否则每个 worker 都会复制一份模型。

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
| LangChain Agent | `http://127.0.0.1:8002` | 对外 Agent 接口 |

浏览器打开 `http://127.0.0.1:8002/` 即可使用客服页面。会话记录默认写入
`artifacts/conversations.sqlite3`，刷新页面后仍可通过浏览器保存的 session ID 延续对话。

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

该模式只启动 Mock 后端和 Agent，并把 LangChain 的模型地址设为
`http://127.0.0.1:11434/v1`。它用于先验证 Agent/Tools，不代表已经使用 QLoRA adapter。

## API

### 推理服务

```text
GET  /health
POST /v1/chat/completions
```

首版只支持 `stream: false`。请求最多 32 条消息、默认最多生成 256 token。工具调用会从
Qwen 的 `<tool_call>...</tool_call>` 输出转换为 OpenAI `tool_calls`。服务会同时限制完整
序列化请求大小，并在分词后校验“输入 token + 输出 token”不超过模型上下文窗口。

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

默认检索器是 CPU 上的轻量 BM25，不占用宝贵的 8GB GPU 显存。后续数据量增大后，可以
把 `KnowledgeBase` 替换为独立 embedding/向量库服务，Agent 的工具契约无需改变。

## 配置

复制 `.env.example` 得到 `.env`。重要变量：

```dotenv
INFERENCE_BASE_MODEL_ID=Qwen/Qwen3-1.7B
INFERENCE_ADAPTER_PATH=artifacts/qlora-adapter
AGENT_MODEL_BASE_URL=http://127.0.0.1:8000/v1
AGENT_MODEL_NAME=local-qlora
AGENT_MOCK_API_URL=http://127.0.0.1:8001
AGENT_KNOWLEDGE_DIR=data/knowledge
AGENT_CONVERSATION_DB_PATH=artifacts/conversations.sqlite3
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

## License

MIT
