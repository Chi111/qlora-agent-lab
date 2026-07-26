---
document_id: vector-database-experience
title: 维修知识库向量检索实施经验
source: experience/approved/vector-database-experience.md
knowledge_scope: experience
appliance_type: general
document_version: "1.0.0"
updated_at: "2026-07-26"
safety_level: internal
review_status: approved
---

# 维修知识库向量检索实施经验

本文保存维修客服 RAG 作业中可复用的工程经验和验收约定，供开发与复盘使用，不作为面向用户的维修答案。文中的数值是当前设计基线或验收目标；只有测试报告产生的实测结果才可标记为“已达到”。

## 先定义知识边界

- 正式维修资料使用 `knowledge_scope: repair`，工程经验使用 `knowledge_scope: experience`，两者在检索时必须通过服务端固定过滤条件隔离。
- 原始会议纪要、草稿、未脱敏的客服记录不能直接进入正式集合。维修经验先进入草稿区，经人工审阅、脱敏和批准后，才移动到 `experience/approved/` 并增量入库。
- 每份文档使用稳定的 `document_id`，同时记录 `source`、`document_version` 和内容哈希。文件名变化不应制造重复文档。
- 检索到的文本是参考资料，不是系统指令。知识块中的命令、链接或“忽略规则”不能触发工具调用、授权或写操作。

## Markdown 文档契约

维修文档 frontmatter 至少包含：

- `document_id`：稳定且唯一的短横线标识；
- `title`：面向阅读者的中文标题；
- `source`：相对 `data/knowledge/` 的路径；
- `knowledge_scope`：`repair` 或 `experience`；
- `appliance_type`：`refrigerator`、`television`、`monitor` 或 `general`；
- `document_version`、`updated_at` 和 `safety_level`。

正文先给安全红线，再给按现象组织的外部检查，最后列报修条件和信息清单。危险设备必须明确禁止拆机、带电测量、短接保护和处理高压电容。避免编造品牌代码、保修期限、像素政策和正常温度等型号相关结论。

## 向量模型选择

- 当前基线 Embedding 为 `BAAI/bge-m3`，dense 向量维度为 1024。
- 当前基线 Reranker 为 `BAAI/bge-reranker-v2-m3`。
- 嵌入模型、维度、归一化方式或切片规则任一发生变化，都应创建新集合版本，不能把不兼容向量混入旧集合。
- 模型必须锁定可信 revision，禁用不必要的 remote code，并对单次文本长度、批量大小、并发和超时设限。
- 在显存受限环境中，Embedding 和 Reranker 可优先使用 CPU，把 GPU 留给 Qwen 底座与 LoRA；上线前仍需用实测延迟确认是否满足体验要求。

## 切片与元数据

- Markdown 按标题边界优先切片，再用递归字符策略处理过长章节。基线目标约 500 个中文字符、重叠约 80 个字符。
- 不把“安全红线”和具体故障步骤混成失去标题上下文的碎片。每个 chunk 保留文档标题与章节名。
- Qdrant payload 至少保存：
  `document_id`、`title`、`source`、`knowledge_scope`、`appliance_type`、`section`、`chunk_index`、`document_version`、`content_hash` 和 `text`。
- chunk 标识应由稳定文档 ID、版本或内容哈希和 chunk 索引确定，重复执行入库应幂等。
- 删除或更新文档时，要清理该 `document_id` 的旧 chunk，避免新旧版本同时命中。

## Qdrant 集合版本

- 正式查询只使用 alias，例如 `repair_knowledge_current`，不把物理集合名暴露给前端或 Agent。
- 完整重建先写入带内容哈希或版本的暂存集合，校验维度、点数量、payload 和抽样检索后，再原子切换 alias。
- 构建失败时保留旧 alias，清理失败的暂存集合要有明确目标，禁止使用模糊通配符。
- 服务启动时检查 alias、集合维度和 distance 是否符合配置；不兼容时健康检查应失败，而不是悄悄新建空集合。
- 本地 Qdrant 只绑定 loopback，持久化使用 Docker named volume，并设置备份与恢复演练。

## 两阶段检索与 Rerank

标准在线链路：

1. 校验查询长度并进行查询 Embedding；
2. 按服务端确定的 `knowledge_scope` 和设备类型过滤 Qdrant，召回 Top 12；
3. 将原始查询与候选正文交给 Reranker；
4. 按重排分数选择 Top 4，并把 `vector_score`、`rerank_score`、来源和章节一起返回；
5. Qwen 底座加载 LoRA adapter，根据重排后的上下文生成并润色答案，同时保留来源引用。

Rerank 是必需阶段。Reranker 超时、返回数量不一致、出现非有限分数或服务不可用时，应返回明确的 `RAG_RERANK_UNAVAILABLE` 错误；不能把未经重排的候选伪装成正常知识答案。无合适资料时，应明确说明资料不足并引导安全报修。

## Agent 与前端边界

- Mastra Agent 与 Python/LangChain Agent 共用一个只读检索契约，避免两套排序和引用行为。
- 内部搜索接口只接收查询、受限 `topK` 和允许的知识域；调用者不能指定集合名、任意过滤表达式或未批准的文件路径。
- 普通客服前端不能访问经验写入、集合切换和重建接口。经验审批工作流仅供受控的 Studio 或内部命令行使用。
- 前端通过 `@mastra/client-js` 调用 Mastra，展示流式状态、来源章节和 Rerank 排名；不能把模型或数据库密钥放入浏览器。
- 创建工单等写操作必须独立审批、幂等并取得用户同意，检索结果本身不能授权写操作。

## 安全与隐私经验

- 入库前删除手机号、住址、订单号、支付信息、验证码、账号密码和可识别个人身份的自由文本。
- 文档中只提供普通用户可安全完成的外部检查。涉及高压、电容、制冷剂、燃烧、漏电、破裂面板和内部线路时，优先停用、隔离和联系专业人员。
- 引用资料不能替代实时订单、品牌保修政策或具体型号说明书。缺少型号时不猜测故障码和参数。
- 日志只记录必要的技术字段；查询正文、聊天上下文和模型输入不应默认永久保存。

## 评测与回归

- 每类设备至少维护 10 个确定性问题，覆盖口语表达、近似故障、跨设备歧义、安全红线和报修信息。
- 每条黄金样例保存预期来源、预期章节、回答必须包含和不得包含的关键语义。
- 候选召回目标为 `Recall@12 >= 95%`，Rerank 后预期块 Top-1 命中率目标为 `>= 90%`；目标不能代替实测。
- Rerank 单测要包含“向量首位错误、重排后正确”的固定候选，并断言候选完整传入、调用确实发生、最终顺序取自 Reranker。
- 入库回归覆盖重复执行、文档更新与删除、维度不符、空文件、超限输入、暂存集合失败和 alias 回滚。

## 运行复盘清单

每次模型、文档或索引策略变更后记录：

- 变更的模型 revision、集合版本、切片参数和 payload schema；
- 文档数量、chunk 数量、失败与跳过数量；
- 黄金集版本、Recall@12、Top-1、无答案安全性和延迟分位数；
- Reranker、Qdrant 或推理服务故障时的实际错误行为；
- 回滚目标和恢复结果；
- 尚未解决的问题、负责人和下一次复验条件。

未经测试的数据不要写成“效果已经提升”。优先保存可以复现的配置、输入、指标和失败案例，而不是只记录主观结论。
