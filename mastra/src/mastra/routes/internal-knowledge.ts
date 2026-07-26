import { timingSafeEqual } from "node:crypto";

import type { ApiRoute } from "@mastra/core/server";
import { z } from "zod";

import { searchKnowledge } from "../rag/knowledge.js";

const requestSchema = z
  .object({
    query: z.string().trim().min(1).max(500),
    topK: z.number().int().min(1).max(8).optional(),
    knowledgeScope: z
      .enum(["repair", "experience", "policy", "product"])
      .optional(),
  })
  .strict();

function secretsEqual(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return (
    leftBuffer.length === rightBuffer.length &&
    timingSafeEqual(leftBuffer, rightBuffer)
  );
}

export async function handleInternalKnowledgeSearch(request: Request): Promise<Response> {
  const expectedSecret = process.env.MASTRA_INTERNAL_SEARCH_KEY;
  if (!expectedSecret) {
    return Response.json(
      { ok: false, error: { code: "INTERNAL_SEARCH_NOT_CONFIGURED" } },
      { status: 503 },
    );
  }
  const suppliedSecret = request.headers.get("x-internal-api-key") ?? "";
  if (!secretsEqual(suppliedSecret, expectedSecret)) {
    return Response.json(
      { ok: false, error: { code: "INTERNAL_SEARCH_UNAUTHORIZED" } },
      { status: 401 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json(
      { ok: false, error: { code: "INVALID_JSON" } },
      { status: 400 },
    );
  }
  const parsed = requestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json(
      { ok: false, error: { code: "INVALID_SEARCH_REQUEST" } },
      { status: 400 },
    );
  }
  const result = await searchKnowledge(parsed.data.query, {
    ...(parsed.data.topK ? { topK: parsed.data.topK } : {}),
    ...(parsed.data.knowledgeScope
      ? { knowledgeScope: parsed.data.knowledgeScope }
      : {}),
  });
  return Response.json(result, { status: result.error ? 503 : 200 });
}

export const internalKnowledgeSearchRoute: ApiRoute = {
  path: "/internal/knowledge/search",
  method: "POST",
  requiresAuth: false,
  handler: async (context: { req: { raw: Request } }) =>
    handleInternalKnowledgeSearch(context.req.raw),
};
