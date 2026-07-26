import { createTool } from "@mastra/core/tools";
import { z } from "zod";

import { searchKnowledge } from "../rag/knowledge.js";

export { searchKnowledge };
export type {
  KnowledgeHit as KnowledgeSearchResult,
  SearchKnowledgeOptions,
  SearchResponse,
} from "../rag/knowledge.js";

const knowledgeResultSchema = z.object({
  document_id: z.string(),
  title: z.string(),
  source: z.string(),
  section: z.string(),
  content: z.string(),
  rank: z.number().int().positive(),
  score: z.number(),
  vector_score: z.number(),
  rerank_score: z.number(),
});

export const knowledgeSearchResponseSchema = z.object({
  ok: z.boolean(),
  reranked: z.boolean(),
  candidate_count: z.number().int().nonnegative(),
  results: z.array(knowledgeResultSchema),
  message: z.string().nullable(),
  error: z
    .object({
      code: z.string(),
      retryable: z.boolean(),
    })
    .optional(),
});

export const searchKnowledgeTool = createTool({
  id: "search_knowledge",
  description:
    "查询冰箱、彩电、显示器维修资料以及产品、配送、退换货等静态知识。所有候选均经过向量召回和 Rerank；不要用于查询具体订单状态。",
  inputSchema: z.object({
    query: z.string().trim().min(1).max(500).describe("需要从知识库检索的问题。"),
  }),
  outputSchema: knowledgeSearchResponseSchema,
  execute: async ({ query }) =>
    searchKnowledge(query, { knowledgeScope: "repair" }),
});
