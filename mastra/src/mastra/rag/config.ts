import path from "node:path";

export const EMBEDDING_DIMENSION = 1_024;
export const VECTOR_CANDIDATE_COUNT = 12;
export const DEFAULT_RESULT_COUNT = 4;
export const MAX_RESULT_COUNT = 8;
export const MAX_QUERY_CHARS = 500;
export const MAX_RERANK_DOCUMENT_CHARS = 12_000;

function numberSetting(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerSetting(value: string | undefined, fallback: number): number {
  const parsed = Math.trunc(numberSetting(value, fallback));
  return parsed > 0 ? parsed : fallback;
}

export const ragConfig = {
  modelServiceUrl: process.env.RAG_MODEL_SERVICE_URL ?? "http://127.0.0.1:8003",
  embeddingModel: process.env.RAG_EMBEDDING_MODEL ?? "BAAI/bge-m3",
  rerankerModel: process.env.RAG_RERANKER_MODEL ?? "BAAI/bge-reranker-v2-m3",
  qdrantUrl: process.env.QDRANT_URL ?? "http://127.0.0.1:6333",
  collection: process.env.QDRANT_COLLECTION ?? "repair_knowledge_current",
  requestTimeoutMs: positiveIntegerSetting(
    process.env.RAG_REQUEST_TIMEOUT_MS,
    15_000,
  ),
  minRerankScore: Math.min(
    Math.max(numberSetting(process.env.RAG_MIN_RERANK_SCORE, 0.2), 0),
    1,
  ),
  knowledgeDirectory:
    process.env.KNOWLEDGE_DIR ?? path.resolve(process.cwd(), "../data/knowledge"),
  experienceDirectory:
    process.env.REPAIR_EXPERIENCE_DIR ??
    path.resolve(process.cwd(), "../data/knowledge/experience"),
} as const;
