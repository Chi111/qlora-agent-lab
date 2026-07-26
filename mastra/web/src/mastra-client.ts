import { MastraClient } from "@mastra/client-js";

import type {
  ChatMessage,
  ConversationContext,
  CustomerServiceTransport,
  KnowledgeSource,
  PendingApproval,
  StreamHandlers,
  ToolEvent,
} from "./types";

const AGENT_ID = "customer-service-agent";

interface StreamChunk {
  type: string;
  runId?: string;
  payload?: Record<string, unknown>;
}

interface ProcessableResponse {
  processDataStream(options: {
    onChunk(chunk: unknown): void | Promise<void>;
  }): Promise<void>;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : undefined;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function toolLabel(name: string, status: ToolEvent["status"]): string {
  const toolNames: Record<string, string> = {
    search_knowledge: "检索维修知识",
    get_order: "查询订单",
    create_ticket: "创建客服工单",
  };
  const statusNames: Record<ToolEvent["status"], string> = {
    running: "执行中",
    complete: "已完成",
    error: "执行失败",
    approval: "等待确认",
  };
  return `${toolNames[name] ?? name} · ${statusNames[status]}`;
}

export function toolResultHasError(payload: Record<string, unknown>): boolean {
  const result = asRecord(payload.result);
  return (
    payload.isError === true ||
    result?.ok === false ||
    (result?.error !== undefined && result.error !== null)
  );
}

function parseSources(result: unknown): KnowledgeSource[] {
  const root = asRecord(result);
  const possibleResults = root?.results;
  if (!Array.isArray(possibleResults)) {
    return [];
  }

  return possibleResults.flatMap((value) => {
    const item = asRecord(value);
    if (!item) {
      return [];
    }
    return [{
      documentId: asString(item.document_id),
      title: asString(item.title) || "未命名资料",
      source: asString(item.source),
      section: asString(item.section),
      content: asString(item.content),
      rank: asNumber(item.rank),
      vectorScore: asNumber(item.vector_score),
      rerankScore: asNumber(item.rerank_score),
    }];
  });
}

function messageText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        const value = asRecord(part);
        return asString(value?.text) || asString(value?.content);
      })
      .filter(Boolean)
      .join("");
  }
  const value = asRecord(content);
  return asString(value?.text) || asString(value?.content);
}

export class MastraCustomerServiceClient implements CustomerServiceTransport {
  private readonly client: MastraClient;
  private readonly agent;

  constructor(baseUrl = import.meta.env.VITE_MASTRA_URL || "http://127.0.0.1:4111") {
    this.client = new MastraClient({ baseUrl });
    this.agent = this.client.getAgent(AGENT_ID);
  }

  async send(
    message: string,
    context: ConversationContext,
    handlers: StreamHandlers,
  ): Promise<void> {
    try {
      const response = await this.agent.stream(
        [{ role: "user", content: message }],
        {
          memory: {
            thread: context.threadId,
            resource: context.resourceId,
          },
          maxSteps: 8,
        },
      );
      await this.consume(response, handlers);
    } catch (error) {
      this.reportError(error, handlers);
    }
  }

  async approve(
    approval: PendingApproval,
    handlers: StreamHandlers,
  ): Promise<void> {
    try {
      const response = await this.agent.approveToolCall({
        runId: approval.runId,
        toolCallId: approval.toolCallId,
      });
      await this.consume(response, handlers);
    } catch (error) {
      this.reportError(error, handlers);
    }
  }

  async decline(
    approval: PendingApproval,
    handlers: StreamHandlers,
  ): Promise<void> {
    try {
      const response = await this.agent.declineToolCall({
        runId: approval.runId,
        toolCallId: approval.toolCallId,
      });
      await this.consume(response, handlers);
    } catch (error) {
      this.reportError(error, handlers);
    }
  }

  async loadHistory(threadId: string): Promise<ChatMessage[]> {
    try {
      const response = await this.client.listThreadMessages(threadId, {
        agentId: AGENT_ID,
      });
      const root = asRecord(response);
      const messages = Array.isArray(root?.messages) ? root.messages : [];
      return messages.flatMap((raw, index) => {
        const message = asRecord(raw);
        const role = message?.role;
        if (!message || (role !== "user" && role !== "assistant")) {
          return [];
        }
        const content = messageText(message.content);
        if (!content) {
          return [];
        }
        return [{
          id: asString(message.id) || `history-${index}`,
          role,
          content,
          sources: [],
          tools: [],
        }];
      });
    } catch {
      return [];
    }
  }

  private async consume(
    response: ProcessableResponse,
    handlers: StreamHandlers,
  ): Promise<void> {
    try {
      await response.processDataStream({
        onChunk: (rawChunk) => {
          const chunk = rawChunk as StreamChunk;
          const payload = chunk.payload ?? {};
          if (chunk.type === "text-delta") {
            handlers.onText(asString(payload.text));
            return;
          }
          if (chunk.type === "tool-call") {
            const toolCallId = asString(payload.toolCallId);
            const toolName = asString(payload.toolName);
            handlers.onToolStart({
              id: toolCallId,
              name: toolName,
              status: "running",
              label: toolLabel(toolName, "running"),
            });
            return;
          }
          if (chunk.type === "tool-result") {
            const toolCallId = asString(payload.toolCallId);
            const toolName = asString(payload.toolName);
            const isError = toolResultHasError(payload);
            const status = isError ? "error" : "complete";
            handlers.onToolResult(
              {
                id: toolCallId,
                name: toolName,
                status,
                label: toolLabel(toolName, status),
              },
              toolName === "search_knowledge"
                ? parseSources(payload.result)
                : [],
            );
            return;
          }
          if (chunk.type === "tool-call-approval") {
            const toolCallId = asString(payload.toolCallId);
            const toolName = asString(payload.toolName);
            const runId = asString(chunk.runId);
            if (runId && toolCallId) {
              handlers.onApproval({
                runId,
                toolCallId,
                toolName,
                args: payload.args,
              });
            }
            return;
          }
          if (chunk.type === "error") {
            const error = asRecord(payload.error);
            handlers.onError(
              asString(error?.message) ||
                asString(payload.message) ||
                "客服暂时无法完成回答，请稍后重试。",
            );
          }
        },
      });
      handlers.onDone();
    } catch (error) {
      handlers.onError(
        error instanceof Error
          ? error.message
          : "无法连接 Mastra 服务，请检查本地服务是否启动。",
      );
    }
  }

  private reportError(error: unknown, handlers: StreamHandlers): void {
    const unavailable =
      error instanceof TypeError && error.message === "Failed to fetch";
    handlers.onError(
      unavailable
        ? "无法连接本地 Mastra 服务，请确认 4111 端口已启动。"
        : error instanceof Error
        ? error.message
        : "无法连接 Mastra 服务，请检查本地服务是否启动。",
    );
  }
}
