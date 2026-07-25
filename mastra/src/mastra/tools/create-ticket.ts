import { createHash } from "node:crypto";

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

import { type FetchOptions, requestJson, type ToolResult } from "./http.js";

const prioritySchema = z.enum(["low", "normal", "high"]);
const ticketSchema = z.object({
  ticket_id: z.string(),
  order_id: z.string(),
  priority: prioritySchema,
  status: z.enum(["open", "closed"]),
  created_at: z.string(),
});

export type SafeTicket = z.infer<typeof ticketSchema>;

export function ticketIdempotencyKey(
  orderId: string,
  reason: string,
  priority: string,
): string {
  const digest = createHash("sha256")
    .update(`${orderId.toUpperCase()}\0${reason}\0${priority}`)
    .digest("hex");
  return `mastra-${digest}`;
}

export async function createTicket(
  orderId: string,
  reason: string,
  priority: "low" | "normal" | "high",
  options: FetchOptions = {},
): Promise<ToolResult<SafeTicket>> {
  const result = await requestJson(
    "/api/tickets",
    {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "Idempotency-Key": ticketIdempotencyKey(orderId, reason, priority),
      },
      body: JSON.stringify({
        order_id: orderId.toUpperCase(),
        reason,
        priority,
      }),
    },
    options,
  );
  if (!result.ok) {
    return {
      ok: false,
      ...(result.status === undefined ? {} : { status_code: result.status }),
      error: result.error,
    };
  }

  const parsed = ticketSchema.safeParse(result.payload);
  if (!parsed.success) {
    return {
      ok: false,
      error: {
        code: "INVALID_BACKEND_RESPONSE",
        message: "Mock backend returned an invalid ticket payload.",
        retryable: false,
      },
    };
  }
  return { ok: true, data: parsed.data };
}

export const createTicketTool = createTool({
  id: "create_ticket",
  description:
    "为已存在的异常订单创建客服工单。只有用户明确要求或同意创建时才能使用；执行前还需要人工批准。",
  requireApproval: true,
  inputSchema: z.object({
    order_id: z
      .string()
      .min(1)
      .max(64)
      .regex(/^[A-Za-z0-9_-]+$/)
      .describe("需要处理的订单号。"),
    reason: z.string().min(3).max(500).describe("创建工单的明确原因。"),
    priority: prioritySchema.default("normal").describe("工单优先级。"),
  }),
  outputSchema: z.union([
    z.object({ ok: z.literal(true), data: ticketSchema }),
    z.object({
      ok: z.literal(false),
      status_code: z.number().optional(),
      error: z.object({
        code: z.string(),
        message: z.string(),
        retryable: z.boolean(),
      }),
    }),
  ]),
  execute: async ({ order_id, reason, priority }) =>
    createTicket(order_id, reason, priority),
});
