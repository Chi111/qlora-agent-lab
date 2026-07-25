import { createTool } from "@mastra/core/tools";
import { z } from "zod";

import { type FetchOptions, requestJson, type ToolResult } from "./http.js";

const orderIdSchema = z
  .string()
  .min(1)
  .max(64)
  .regex(/^[A-Za-z0-9_-]+$/);

const backendOrderSchema = z.object({
  order_id: z.string(),
  status: z.enum(["paid", "processing", "shipped", "delayed", "cancelled"]),
  item: z.string(),
  updated_at: z.string(),
});

export type SafeOrder = z.infer<typeof backendOrderSchema>;

export async function getOrder(
  orderId: string,
  options: FetchOptions = {},
): Promise<ToolResult<SafeOrder>> {
  const result = await requestJson(
    `/api/orders/${encodeURIComponent(orderId.toUpperCase())}`,
    { method: "GET", headers: { Accept: "application/json" } },
    options,
  );
  if (!result.ok) {
    return {
      ok: false,
      ...(result.status === undefined ? {} : { status_code: result.status }),
      error: result.error,
    };
  }

  const parsed = backendOrderSchema.safeParse(result.payload);
  if (!parsed.success) {
    return {
      ok: false,
      error: {
        code: "INVALID_BACKEND_RESPONSE",
        message: "Mock backend returned an invalid order payload.",
        retryable: false,
      },
    };
  }
  return { ok: true, data: parsed.data };
}

export const getOrderTool = createTool({
  id: "get_order",
  description: "查询一个订单的当前状态。用户询问具体订单信息时使用；不要猜测订单状态。",
  inputSchema: z.object({
    order_id: orderIdSchema.describe("订单号，例如 A100。"),
  }),
  outputSchema: z.union([
    z.object({ ok: z.literal(true), data: backendOrderSchema }),
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
  execute: async ({ order_id }) => getOrder(order_id),
});
