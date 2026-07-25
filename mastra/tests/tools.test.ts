import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";

import {
  createTicketTool,
  createTicket,
  ticketIdempotencyKey,
} from "../src/mastra/tools/create-ticket.js";
import { getOrder } from "../src/mastra/tools/get-order.js";
import { searchKnowledge } from "../src/mastra/tools/search-knowledge.js";

describe("getOrder", () => {
  it("returns only the fields safe for model context", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({
          order_id: "A100",
          customer_name: "不应泄露",
          status: "shipped",
          item: "机械键盘",
          amount_cny: 399,
          updated_at: "2026-07-25T12:00:00Z",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const result = await getOrder("a100", {
      baseUrl: "http://mock.invalid",
      fetchImpl,
    });

    expect(result).toEqual({
      ok: true,
      data: {
        order_id: "A100",
        status: "shipped",
        item: "机械键盘",
        updated_at: "2026-07-25T12:00:00Z",
      },
    });
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://mock.invalid/api/orders/A100",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns a structured error when reading the response body fails", async () => {
    const fetchImpl = vi.fn(async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.error(new Error("socket closed"));
        },
      });
      return new Response(body, { status: 200 });
    });

    const result = await getOrder("A100", {
      baseUrl: "http://mock.invalid",
      fetchImpl,
    });

    expect(result).toEqual({
      ok: false,
      status_code: 200,
      error: {
        code: "BACKEND_RESPONSE_READ_FAILED",
        message: "socket closed",
        retryable: true,
      },
    });
  });
});

describe("createTicket", () => {
  it("requires runtime approval before execution", () => {
    expect(createTicketTool.requireApproval).toBe(true);
  });

  it("uses a stable idempotency key and removes the reason from tool output", async () => {
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect((init?.headers as Record<string, string>)["Idempotency-Key"]).toBe(
        ticketIdempotencyKey("A101", "物流延迟", "high"),
      );
      return new Response(
        JSON.stringify({
          ticket_id: "T-1",
          order_id: "A101",
          reason: "物流延迟",
          priority: "high",
          status: "open",
          created_at: "2026-07-25T12:00:00Z",
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      );
    });

    const result = await createTicket("A101", "物流延迟", "high", {
      baseUrl: "http://mock.invalid",
      fetchImpl,
    });

    expect(result).toEqual({
      ok: true,
      data: {
        ticket_id: "T-1",
        order_id: "A101",
        priority: "high",
        status: "open",
        created_at: "2026-07-25T12:00:00Z",
      },
    });
  });
});

describe("searchKnowledge", () => {
  it("finds a Chinese policy document and includes its source", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-knowledge-"));
    await writeFile(
      path.join(directory, "returns.md"),
      "# 退换货政策\n签收后七天内，商品完好可以申请退货。",
      "utf8",
    );
    await writeFile(
      path.join(directory, "delivery.md"),
      "# 配送政策\n普通快递预计三到五天送达。",
      "utf8",
    );

    const results = await searchKnowledge("七天内可以退货吗", directory, 1);

    expect(results).toHaveLength(1);
    expect(results[0]?.source).toBe("returns.md");
    expect(results[0]?.content).toContain("七天内");
  });
});
