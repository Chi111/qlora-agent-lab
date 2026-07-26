import {
  EMBEDDING_DIMENSION,
  MAX_RERANK_DOCUMENT_CHARS,
  ragConfig,
} from "./config.js";
import { RagServiceError, requestJson } from "./http.js";

interface EmbeddingResponse {
  data?: Array<{ index?: number; embedding?: number[] }>;
}

interface RerankItem {
  index?: number;
  relevance_score?: number;
  score?: number;
}

interface RerankResponse {
  results?: RerankItem[];
  data?: RerankItem[];
}

export interface ModelServiceOptions {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export async function embedTexts(
  texts: string[],
  options: ModelServiceOptions = {},
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const baseUrl = (options.baseUrl ?? ragConfig.modelServiceUrl).replace(/\/+$/, "");
  const response = await requestJson<EmbeddingResponse>(
    `${baseUrl}/v1/embeddings`,
    {
      method: "POST",
      body: JSON.stringify({
        model: ragConfig.embeddingModel,
        input: texts,
        encoding_format: "float",
      }),
    },
    {
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      timeoutMs: options.timeoutMs ?? ragConfig.requestTimeoutMs,
    },
  );

  const ordered = [...(response.data ?? [])].sort(
    (left, right) => (left.index ?? 0) - (right.index ?? 0),
  );
  if (
    ordered.length !== texts.length ||
    ordered.some(
      (item) =>
        !Array.isArray(item.embedding) || item.embedding.length !== EMBEDDING_DIMENSION,
    )
  ) {
    throw new RagServiceError(
      "RAG_EMBEDDING_INVALID",
      `Embedding 服务必须返回 ${texts.length} 个 ${EMBEDDING_DIMENSION} 维向量。`,
      { retryable: false },
    );
  }
  return ordered.map((item) => item.embedding as number[]);
}

export interface RerankScore {
  index: number;
  score: number;
}

export async function rerankDocuments(
  query: string,
  documents: string[],
  topN: number,
  options: ModelServiceOptions = {},
): Promise<RerankScore[]> {
  if (documents.length === 0) return [];
  const baseUrl = (options.baseUrl ?? ragConfig.modelServiceUrl).replace(/\/+$/, "");
  const response = await requestJson<RerankResponse>(
    `${baseUrl}/v1/rerank`,
    {
      method: "POST",
      body: JSON.stringify({
        model: ragConfig.rerankerModel,
        query,
        documents: documents.map((document) =>
          document.slice(0, MAX_RERANK_DOCUMENT_CHARS),
        ),
        top_n: Math.min(Math.max(topN, 1), documents.length),
      }),
    },
    {
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      timeoutMs: options.timeoutMs ?? ragConfig.requestTimeoutMs,
    },
  );

  const rows = response.results ?? response.data ?? [];
  const parsed = rows
    .map((item) => ({
      index: item.index,
      score: item.relevance_score ?? item.score,
    }))
    .filter(
      (item): item is RerankScore =>
        Number.isInteger(item.index) &&
        (item.index as number) >= 0 &&
        (item.index as number) < documents.length &&
        Number.isFinite(item.score),
    );
  const unique = new Set(parsed.map(({ index }) => index));
  const expectedResultCount = Math.min(Math.max(topN, 1), documents.length);
  if (
    parsed.length !== expectedResultCount ||
    unique.size !== parsed.length
  ) {
    throw new RagServiceError(
      "RAG_RERANK_INVALID",
      `Rerank 服务必须返回 ${expectedResultCount} 条不重复的有效排名。`,
      { retryable: false },
    );
  }
  return parsed
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(Math.max(topN, 1), documents.length));
}
