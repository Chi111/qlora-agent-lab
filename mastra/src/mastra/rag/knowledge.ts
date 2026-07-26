import type { QueryResult } from "@mastra/core/vector";

import {
  DEFAULT_RESULT_COUNT,
  MAX_QUERY_CHARS,
  MAX_RESULT_COUNT,
  ragConfig,
  VECTOR_CANDIDATE_COUNT,
} from "./config.js";
import { RagServiceError } from "./http.js";
import { embedTexts, rerankDocuments } from "./model-service.js";
import type { RerankScore } from "./model-service.js";
import { qdrantVector } from "../vectors/qdrant.js";
import type { QdrantFilter } from "../vectors/qdrant.js";

export interface KnowledgeHit {
  document_id: string;
  title: string;
  source: string;
  section: string;
  content: string;
  rank: number;
  score: number;
  vector_score: number;
  rerank_score: number;
}

export interface SearchResponse {
  ok: boolean;
  reranked: boolean;
  candidate_count: number;
  results: KnowledgeHit[];
  message: string | null;
  error?: { code: string; retryable: boolean };
}

export interface KnowledgeVectorStore {
  query(params: {
    indexName: string;
    queryVector: number[];
    topK: number;
    filter?: QdrantFilter;
  }): Promise<QueryResult[]>;
}

export interface SearchKnowledgeOptions {
  topK?: number;
  knowledgeScope?: string;
  collection?: string;
  vectorStore?: KnowledgeVectorStore;
  embed?: (texts: string[]) => Promise<number[][]>;
  rerank?: (
    query: string,
    documents: string[],
    topN: number,
  ) => Promise<RerankScore[]>;
  minRerankScore?: number;
}

function stringMetadata(metadata: Record<string, unknown>, key: string): string {
  const value = metadata[key];
  return typeof value === "string" ? value : "";
}

function safeScore(value: number): number {
  return Number(Number.isFinite(value) ? value.toFixed(6) : 0);
}

function candidateContent(candidate: QueryResult): string {
  const metadata = candidate.metadata ?? {};
  return stringMetadata(metadata, "text") || candidate.document || "";
}

function mapHit(candidate: QueryResult, rerank: RerankScore, rank: number): KnowledgeHit {
  const metadata = candidate.metadata ?? {};
  const rerankScore = safeScore(rerank.score);
  return {
    document_id: stringMetadata(metadata, "document_id") || candidate.id,
    title: stringMetadata(metadata, "title") || "未命名资料",
    source: stringMetadata(metadata, "source") || "unknown",
    section: stringMetadata(metadata, "section"),
    content: candidateContent(candidate),
    rank,
    score: rerankScore,
    vector_score: safeScore(candidate.score),
    rerank_score: rerankScore,
  };
}

function failure(code: string, retryable: boolean, message: string): SearchResponse {
  return {
    ok: false,
    reranked: false,
    candidate_count: 0,
    results: [],
    message,
    error: { code, retryable },
  };
}

export async function searchKnowledge(
  query: string,
  options: SearchKnowledgeOptions = {},
): Promise<SearchResponse> {
  const normalizedQuery = query.trim();
  if (!normalizedQuery || normalizedQuery.length > MAX_QUERY_CHARS) {
    return failure("RAG_INVALID_QUERY", false, "查询内容必须为 1 到 500 个字符。");
  }

  const resultCount = Math.min(
    Math.max(options.topK ?? DEFAULT_RESULT_COUNT, 1),
    MAX_RESULT_COUNT,
  );
  let queryVector: number[];
  try {
    const embeddings = await (options.embed ?? embedTexts)([normalizedQuery]);
    const first = embeddings[0];
    if (!first) {
      throw new RagServiceError("RAG_EMBEDDING_INVALID", "Embedding 结果为空。");
    }
    queryVector = first;
  } catch (error) {
    const ragError = error instanceof RagServiceError ? error : undefined;
    return failure(
      "RAG_EMBEDDING_UNAVAILABLE",
      ragError?.retryable ?? true,
      "查询向量模型暂时不可用。",
    );
  }

  let candidates: QueryResult[];
  try {
    const filter = options.knowledgeScope
      ? { knowledge_scope: options.knowledgeScope }
      : undefined;
    candidates = await (options.vectorStore ?? qdrantVector).query({
      indexName: options.collection ?? ragConfig.collection,
      queryVector,
      topK: VECTOR_CANDIDATE_COUNT,
      ...(filter ? { filter } : {}),
    });
  } catch (error) {
    const ragError = error instanceof RagServiceError ? error : undefined;
    return failure(
      "RAG_VECTOR_UNAVAILABLE",
      ragError?.retryable ?? true,
      "向量数据库暂时不可用。",
    );
  }

  const usableCandidates = candidates
    .filter((candidate) => candidateContent(candidate).trim().length > 0)
    .slice(0, VECTOR_CANDIDATE_COUNT);
  if (usableCandidates.length === 0) {
    return {
      ok: false,
      reranked: true,
      candidate_count: candidates.length,
      results: [],
      message: "知识库中没有找到相关内容。",
    };
  }

  let reranked: RerankScore[];
  try {
    reranked = await (options.rerank ?? rerankDocuments)(
      normalizedQuery,
      usableCandidates.map(candidateContent),
      resultCount,
    );
  } catch (error) {
    const ragError = error instanceof RagServiceError ? error : undefined;
    return {
      ok: false,
      reranked: false,
      candidate_count: usableCandidates.length,
      results: [],
      message: "候选资料未能完成重排，本次不会使用未经重排的内容回答。",
      error: {
        code: "RAG_RERANK_UNAVAILABLE",
        retryable: ragError?.retryable ?? true,
      },
    };
  }

  const minRerankScore = Math.min(
    Math.max(options.minRerankScore ?? ragConfig.minRerankScore, 0),
    1,
  );
  const results = reranked
    .filter(({ score }) => score >= minRerankScore)
    .map((rerank, index) => {
      const candidate = usableCandidates[rerank.index];
      return candidate ? mapHit(candidate, rerank, index + 1) : undefined;
    })
    .filter((hit): hit is KnowledgeHit => hit !== undefined);
  if (results.length === 0) {
    return {
      ok: false,
      reranked: true,
      candidate_count: usableCandidates.length,
      results: [],
      message: `重排后没有达到相关性阈值 ${minRerankScore} 的知识片段。`,
    };
  }
  return {
    ok: true,
    reranked: true,
    candidate_count: usableCandidates.length,
    results,
    message: null,
  };
}
