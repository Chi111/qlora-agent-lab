export type ChatRole = "user" | "assistant";

export interface KnowledgeSource {
  documentId: string;
  title: string;
  source: string;
  section: string;
  content: string;
  rank: number;
  vectorScore: number;
  rerankScore: number;
}

export interface ToolEvent {
  id: string;
  name: string;
  status: "running" | "complete" | "error" | "approval";
  label: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  sources: KnowledgeSource[];
  tools: ToolEvent[];
}

export interface PendingApproval {
  assistantMessageId: string;
  runId: string;
  toolCallId: string;
  toolName: string;
  args: unknown;
}

export interface StreamHandlers {
  onText(text: string): void;
  onToolStart(event: ToolEvent): void;
  onToolResult(event: ToolEvent, sources: KnowledgeSource[]): void;
  onApproval(approval: Omit<PendingApproval, "assistantMessageId">): void;
  onError(message: string): void;
  onDone(): void;
}

export interface ConversationContext {
  threadId: string;
  resourceId: string;
}

export interface CustomerServiceTransport {
  send(
    message: string,
    context: ConversationContext,
    handlers: StreamHandlers,
  ): Promise<void>;
  approve(
    approval: PendingApproval,
    handlers: StreamHandlers,
  ): Promise<void>;
  decline(
    approval: PendingApproval,
    handlers: StreamHandlers,
  ): Promise<void>;
  loadHistory(threadId: string): Promise<ChatMessage[]>;
}
