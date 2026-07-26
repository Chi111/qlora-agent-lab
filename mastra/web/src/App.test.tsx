import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "./App";
import type {
  ChatMessage,
  ConversationContext,
  CustomerServiceTransport,
  PendingApproval,
  StreamHandlers,
} from "./types";

class FakeTransport implements CustomerServiceTransport {
  send = vi.fn(
    async (
      _message: string,
      _context: ConversationContext,
      handlers: StreamHandlers,
    ) => {
      handlers.onToolStart({
        id: "tool-1",
        name: "search_knowledge",
        status: "running",
        label: "检索维修知识 · 执行中",
      });
      handlers.onToolResult(
        {
          id: "tool-1",
          name: "search_knowledge",
          status: "complete",
          label: "检索维修知识 · 已完成",
        },
        [{
          documentId: "refrigerator-repair",
          title: "冰箱维修指南",
          source: "data/knowledge/refrigerator-repair.md",
          section: "不制冷",
          content: "先确认电源和温控设置，再检查门封。",
          rank: 1,
          vectorScore: 0.76,
          rerankScore: 0.93,
        }],
      );
      handlers.onText("请先断电，再检查电源和温控设置。");
      handlers.onDone();
    },
  );

  approve = vi.fn(async (_approval: PendingApproval, handlers: StreamHandlers) => {
    handlers.onDone();
  });

  decline = vi.fn(async (_approval: PendingApproval, handlers: StreamHandlers) => {
    handlers.onDone();
  });

  loadHistory = vi.fn(async (): Promise<ChatMessage[]> => []);
}

class ApprovalTransport extends FakeTransport {
  send = vi.fn(
    async (
      _message: string,
      _context: ConversationContext,
      handlers: StreamHandlers,
    ) => {
      handlers.onApproval({
        runId: "run-1",
        toolCallId: "tool-approval-1",
        toolName: "create_ticket",
        args: {
          order_id: "A101",
          reason: "物流延迟",
          priority: "high",
        },
      });
      handlers.onDone();
    },
  );
}

describe("customer service app", () => {
  it("streams an answer and exposes reranked sources", async () => {
    const transport = new FakeTransport();
    render(<App transport={transport} />);

    await screen.findByText("冰箱不制冷应该怎么排查？");
    fireEvent.click(screen.getByText("冰箱不制冷应该怎么排查？"));

    await screen.findByText("请先断电，再检查电源和温控设置。");
    expect(transport.send).toHaveBeenCalledOnce();
    expect(screen.getByText("检索维修知识 · 已完成")).toBeInTheDocument();

    fireEvent.click(screen.getByText(/本次引用 1 条重排后的维修资料/));
    expect(screen.getByText("冰箱维修指南")).toBeInTheDocument();
    expect(screen.getByText("重排 0.930")).toBeInTheDocument();
  });

  it("starts a fresh local thread", async () => {
    const transport = new FakeTransport();
    render(<App transport={transport} />);
    await waitFor(() => expect(transport.loadHistory).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /新建会话/ }));
    expect(
      await screen.findByText(/你好，我是启智售后助手/),
    ).toBeInTheDocument();
  });

  it("shows write parameters before approving a tool call", async () => {
    const transport = new ApprovalTransport();
    render(<App transport={transport} />);
    await screen.findByText("冰箱不制冷应该怎么排查？");

    fireEvent.click(screen.getByText("冰箱不制冷应该怎么排查？"));

    expect(await screen.findByText("A101")).toBeInTheDocument();
    expect(screen.getByText("物流延迟")).toBeInTheDocument();
    expect(screen.getByText("high")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "确认执行" }));
    await waitFor(() => expect(transport.approve).toHaveBeenCalledOnce());
  });
});
