import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { MastraCustomerServiceClient } from "./mastra-client";
import type {
  ChatMessage,
  CustomerServiceTransport,
  KnowledgeSource,
  PendingApproval,
  StreamHandlers,
  ToolEvent,
} from "./types";

const THREAD_KEY = "qlora-customer-service-thread";
const RESOURCE_KEY = "qlora-customer-service-resource";
const welcomeMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "你好，我是启智售后助手。可以帮你排查冰箱、彩电和显示器故障，也能查询订单和退换货政策。",
  sources: [],
  tools: [],
};

const quickPrompts = [
  "冰箱不制冷应该怎么排查？",
  "彩电有声音但没有画面怎么办？",
  "显示器频繁黑屏可能是什么原因？",
];

function createId(prefix: string): string {
  const suffix =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${suffix}`;
}

function getOrCreateStorageId(key: string, prefix: string): string {
  const current = window.localStorage.getItem(key);
  if (current) {
    return current;
  }
  const created = createId(prefix);
  window.localStorage.setItem(key, created);
  return created;
}

function score(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function approvalRows(args: unknown): Array<{ label: string; value: string }> {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return [];
  }
  const labels: Record<string, string> = {
    order_id: "订单号",
    orderId: "订单号",
    reason: "原因",
    priority: "优先级",
  };
  return Object.entries(args as Record<string, unknown>)
    .slice(0, 8)
    .map(([key, value]) => {
      const rendered =
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean"
          ? String(value)
          : JSON.stringify(value);
      return {
        label: labels[key] ?? key,
        value: (rendered || "—").slice(0, 500),
      };
    });
}

function ToolPill({ event }: { event: ToolEvent }) {
  return (
    <span className={`tool-pill tool-pill--${event.status}`}>
      <span className="tool-pill__dot" aria-hidden="true" />
      {event.label}
    </span>
  );
}

function SourceCard({ source }: { source: KnowledgeSource }) {
  return (
    <article className="source-card">
      <div className="source-card__head">
        <span className="source-card__rank">#{source.rank || "—"}</span>
        <strong>{source.title}</strong>
      </div>
      <p>{source.section || "维修知识"}</p>
      <blockquote>{source.content}</blockquote>
      <div className="source-card__scores">
        <span>向量 {score(source.vectorScore)}</span>
        <span>重排 {score(source.rerankScore)}</span>
      </div>
    </article>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={`message message--${message.role}`}>
      <div className="message__avatar" aria-hidden="true">
        {isAssistant ? "AI" : "你"}
      </div>
      <div className="message__body">
        <div className="message__label">{isAssistant ? "启智助手" : "我的问题"}</div>
        <div className="message__bubble">
          {message.content ? (
            <p>{message.content}</p>
          ) : (
            <span className="typing" aria-label="正在生成回答">
              <i />
              <i />
              <i />
            </span>
          )}
        </div>
        {message.tools.length > 0 && (
          <div className="tool-list" aria-label="工具执行状态">
            {message.tools.map((event) => (
              <ToolPill key={event.id} event={event} />
            ))}
          </div>
        )}
        {message.sources.length > 0 && (
          <details className="sources">
            <summary>
              本次引用 {message.sources.length} 条重排后的维修资料
            </summary>
            <div className="sources__grid">
              {message.sources.map((source) => (
                <SourceCard
                  key={`${source.documentId}-${source.rank}`}
                  source={source}
                />
              ))}
            </div>
          </details>
        )}
      </div>
    </article>
  );
}

export interface AppProps {
  transport?: CustomerServiceTransport;
}

export default function App({ transport }: AppProps) {
  const api = useMemo(
    () => transport ?? new MastraCustomerServiceClient(),
    [transport],
  );
  const [threadId, setThreadId] = useState(() =>
    getOrCreateStorageId(THREAD_KEY, "thread"),
  );
  const [resourceId] = useState(() =>
    getOrCreateStorageId(RESOURCE_KEY, "visitor"),
  );
  const [messages, setMessages] = useState<ChatMessage[]>([welcomeMessage]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState("");
  const [pendingApproval, setPendingApproval] =
    useState<PendingApproval | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    let restoreTimer: ReturnType<typeof window.setTimeout> | undefined;
    setRestoring(true);
    const restoreTimeout = new Promise<ChatMessage[]>((resolve) => {
      restoreTimer = window.setTimeout(() => resolve([]), 2_000);
    });
    void Promise.race([api.loadHistory(threadId), restoreTimeout]).then((history) => {
      if (!active) {
        return;
      }
      if (restoreTimer !== undefined) {
        window.clearTimeout(restoreTimer);
      }
      setMessages(history.length > 0 ? history : [welcomeMessage]);
      setRestoring(false);
    });
    return () => {
      active = false;
      if (restoreTimer !== undefined) {
        window.clearTimeout(restoreTimer);
      }
    };
  }, [api, threadId]);

  useEffect(() => {
    if (typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, pendingApproval, busy]);

  function updateAssistant(
    assistantId: string,
    update: (message: ChatMessage) => ChatMessage,
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId ? update(message) : message,
      ),
    );
  }

  function handlersFor(assistantId: string): StreamHandlers {
    return {
      onText: (text) => {
        if (!text) {
          return;
        }
        updateAssistant(assistantId, (message) => ({
          ...message,
          content: message.content + text,
        }));
      },
      onToolStart: (event) => {
        updateAssistant(assistantId, (message) => ({
          ...message,
          tools: [
            ...message.tools.filter((item) => item.id !== event.id),
            event,
          ],
        }));
      },
      onToolResult: (event, sources) => {
        updateAssistant(assistantId, (message) => ({
          ...message,
          sources: sources.length > 0 ? sources : message.sources,
          tools: [
            ...message.tools.filter((item) => item.id !== event.id),
            event,
          ],
        }));
      },
      onApproval: (approval) => {
        const event: ToolEvent = {
          id: approval.toolCallId,
          name: approval.toolName,
          status: "approval",
          label: `${approval.toolName} · 等待确认`,
        };
        updateAssistant(assistantId, (message) => ({
          ...message,
          tools: [
            ...message.tools.filter((item) => item.id !== event.id),
            event,
          ],
        }));
        setPendingApproval({ ...approval, assistantMessageId: assistantId });
      },
      onError: (message) => {
        updateAssistant(assistantId, (assistant) => ({
          ...assistant,
          content:
            assistant.content ||
            "暂时无法连接客服服务，请确认本地 Mastra 服务已启动后重试。",
        }));
        setError(message);
        setBusy(false);
      },
      onDone: () => {
        setBusy(false);
      },
    };
  }

  async function submitMessage(text: string) {
    const normalized = text.trim();
    if (!normalized || busy) {
      return;
    }
    setInput("");
    setError("");
    setBusy(true);
    const assistantId = createId("assistant");
    setMessages((current) => [
      ...current,
      {
        id: createId("user"),
        role: "user",
        content: normalized,
        sources: [],
        tools: [],
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        sources: [],
        tools: [],
      },
    ]);
    await api.send(
      normalized,
      { threadId, resourceId },
      handlersFor(assistantId),
    );
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submitMessage(input);
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage(input);
    }
  }

  async function resolveApproval(approved: boolean) {
    if (!pendingApproval || busy) {
      return;
    }
    const approval = pendingApproval;
    setPendingApproval(null);
    setBusy(true);
    const handlers = handlersFor(approval.assistantMessageId);
    if (approved) {
      await api.approve(approval, handlers);
    } else {
      await api.decline(approval, handlers);
    }
  }

  function newConversation() {
    if (busy) {
      return;
    }
    const nextThread = createId("thread");
    window.localStorage.setItem(THREAD_KEY, nextThread);
    setThreadId(nextThread);
    setMessages([welcomeMessage]);
    setPendingApproval(null);
    setError("");
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark">启</div>
          <div>
            <strong>启智售后</strong>
            <span>RAG Customer Care</span>
          </div>
        </div>
        <button
          className="new-chat"
          type="button"
          onClick={newConversation}
          disabled={busy}
        >
          <span aria-hidden="true">＋</span>
          新建会话
        </button>
        <section className="sidebar__section">
          <span className="eyebrow">知识范围</span>
          <ul className="knowledge-list">
            <li><span>冰</span>冰箱维修</li>
            <li><span>彩</span>彩电维修</li>
            <li><span>显</span>显示器维修</li>
          </ul>
        </section>
        <div className="pipeline-card">
          <span className="pipeline-card__status">
            <i aria-hidden="true" />
            本地知识链路
          </span>
          <strong>BGE-M3 → Qdrant → Rerank</strong>
          <small>仅使用重排后的资料生成答案</small>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <span className="eyebrow">AI 售后工作台</span>
            <h1>维修问题，先查证再回答</h1>
          </div>
          <div className="model-badge">
            <span aria-hidden="true">QLoRA</span>
            Qwen + LoRA
          </div>
        </header>

        <div
          className="messages"
          aria-live="polite"
          aria-busy={busy || restoring}
        >
          {restoring ? (
            <div className="restore-state">正在恢复本地会话…</div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}

          {pendingApproval && (
            <section className="approval-card" aria-label="敏感操作确认">
              <div>
                <span className="eyebrow">需要人工确认</span>
                <strong>{pendingApproval.toolName}</strong>
                <p>该工具会改变系统状态。请逐项核对后再继续。</p>
                <dl className="approval-card__details">
                  {approvalRows(pendingApproval.args).map((item) => (
                    <div key={item.label}>
                      <dt>{item.label}</dt>
                      <dd>{item.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              <div className="approval-card__actions">
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={() => void resolveApproval(false)}
                >
                  拒绝
                </button>
                <button
                  type="button"
                  className="button button--primary"
                  onClick={() => void resolveApproval(true)}
                >
                  确认执行
                </button>
              </div>
            </section>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <strong>本次请求未完成</strong>
              <span>{error}</span>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <footer className="composer-wrap">
          {messages.length <= 1 && !restoring && (
            <div className="quick-prompts" aria-label="常见问题">
              {quickPrompts.map((prompt) => (
                <button
                  type="button"
                  key={prompt}
                  onClick={() => void submitMessage(prompt)}
                  disabled={busy}
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}
          <form className="composer" onSubmit={onSubmit}>
            <label htmlFor="question" className="sr-only">输入售后问题</label>
            <textarea
              id="question"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder="描述故障现象、设备型号和已经尝试过的步骤…"
              rows={2}
              maxLength={2000}
              disabled={busy || restoring || pendingApproval !== null}
            />
            <button
              className="send-button"
              type="submit"
              disabled={!input.trim() || busy || restoring || pendingApproval !== null}
              aria-label="发送问题"
            >
              {busy ? <span className="spinner" /> : <span aria-hidden="true">↑</span>}
            </button>
          </form>
          <p className="composer-note">
            AI 建议不能替代持证维修；涉及带电、高压或制冷剂操作请立即停机并联系专业人员。
          </p>
        </footer>
      </section>
    </main>
  );
}
