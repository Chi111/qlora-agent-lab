import { randomUUID } from "node:crypto";

import { MastraVector } from "@mastra/core/vector";
import type {
  CreateIndexParams,
  DeleteIndexParams,
  DeleteVectorParams,
  DeleteVectorsParams,
  DescribeIndexParams,
  IndexStats,
  QueryResult,
  QueryVectorParams,
  UpdateVectorParams,
  UpsertVectorParams,
} from "@mastra/core/vector";

import { ragConfig } from "../rag/config.js";
import { RagServiceError, requestJson } from "../rag/http.js";

export type QdrantFilter = Record<string, unknown>;

interface QdrantResponse<T> {
  result: T;
  status?: string;
}

interface QdrantPoint {
  id: string | number;
  score: number;
  payload?: Record<string, unknown>;
  vector?: number[];
}

interface QdrantAlias {
  alias_name: string;
  collection_name: string;
}

function metricName(metric: CreateIndexParams["metric"]): string {
  if (metric === "euclidean") return "Euclid";
  if (metric === "dotproduct") return "Dot";
  return "Cosine";
}

function mastraMetric(
  metric: string | undefined,
): NonNullable<IndexStats["metric"]> {
  if (metric?.toLowerCase() === "euclid") return "euclidean";
  if (metric?.toLowerCase() === "dot") return "dotproduct";
  return "cosine";
}

function qdrantFilter(filter: QdrantFilter | undefined): Record<string, unknown> | undefined {
  if (!filter || Object.keys(filter).length === 0) return undefined;
  const and = filter.$and;
  const entries =
    Array.isArray(and) && Object.keys(filter).length === 1
      ? and.flatMap((item) =>
          item && typeof item === "object" ? Object.entries(item as Record<string, unknown>) : [],
        )
      : Object.entries(filter);
  return {
    must: entries
      .filter(([key]) => !key.startsWith("$"))
      .map(([key, value]) => ({ key, match: { value } })),
  };
}

export interface QdrantVectorOptions {
  id?: string;
  url?: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export class QdrantVectorStore extends MastraVector<QdrantFilter> {
  private readonly url: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch | undefined;
  private readonly timeoutMs: number;

  constructor(options: QdrantVectorOptions = {}) {
    super({ id: options.id ?? "qdrant-repair-knowledge" });
    this.url = (options.url ?? ragConfig.qdrantUrl).replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? process.env.QDRANT_API_KEY;
    this.fetchImpl = options.fetchImpl;
    this.timeoutMs = options.timeoutMs ?? ragConfig.requestTimeoutMs;
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const headers = this.apiKey ? { "api-key": this.apiKey } : undefined;
    const response = await requestJson<QdrantResponse<T>>(
      `${this.url}${path}`,
      init,
      {
        ...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
        timeoutMs: this.timeoutMs,
        ...(headers ? { headers } : {}),
      },
    );
    return response.result;
  }

  async query(params: QueryVectorParams<QdrantFilter>): Promise<QueryResult[]> {
    if (!params.queryVector && !params.filter) {
      throw new RagServiceError("RAG_INVALID_QUERY", "向量查询必须包含向量或过滤条件。");
    }
    if (!params.queryVector) {
      const points = await this.request<{ points: Array<Omit<QdrantPoint, "score">> }>(
        `/collections/${encodeURIComponent(params.indexName)}/points/scroll`,
        {
          method: "POST",
          body: JSON.stringify({
            filter: qdrantFilter(params.filter),
            limit: params.topK ?? 10,
            with_payload: true,
            with_vector: params.includeVector ?? false,
          }),
        },
      );
      return points.points.map((point) => ({
        id: String(point.id),
        score: 1,
        ...(point.payload ? { metadata: point.payload } : {}),
        ...(point.vector ? { vector: point.vector } : {}),
      }));
    }

    const points = await this.request<QdrantPoint[]>(
      `/collections/${encodeURIComponent(params.indexName)}/points/search`,
      {
        method: "POST",
        body: JSON.stringify({
          vector: params.queryVector,
          limit: params.topK ?? 10,
          filter: qdrantFilter(params.filter),
          with_payload: true,
          with_vector: params.includeVector ?? false,
        }),
      },
    );
    return points.map((point) => ({
      id: String(point.id),
      score: point.score,
      ...(point.payload ? { metadata: point.payload } : {}),
      ...(point.vector ? { vector: point.vector } : {}),
    }));
  }

  async upsert(params: UpsertVectorParams): Promise<string[]> {
    if (
      params.metadata !== undefined &&
      params.metadata.length !== params.vectors.length
    ) {
      throw new RagServiceError("RAG_INVALID_UPSERT", "向量和 metadata 数量不一致。");
    }
    if (params.ids !== undefined && params.ids.length !== params.vectors.length) {
      throw new RagServiceError("RAG_INVALID_UPSERT", "向量和 ID 数量不一致。");
    }
    if (params.deleteFilter) {
      await this.deleteVectors({ indexName: params.indexName, filter: params.deleteFilter });
    }
    const ids = params.ids ?? params.vectors.map(() => randomUUID());
    await this.request(
      `/collections/${encodeURIComponent(params.indexName)}/points?wait=true`,
      {
        method: "PUT",
        body: JSON.stringify({
          points: params.vectors.map((vector, index) => ({
            id: ids[index],
            vector,
            payload: params.metadata?.[index] ?? {},
          })),
        }),
      },
    );
    return ids;
  }

  async createIndex(params: CreateIndexParams): Promise<void> {
    await this.request(`/collections/${encodeURIComponent(params.indexName)}`, {
      method: "PUT",
      body: JSON.stringify({
        vectors: { size: params.dimension, distance: metricName(params.metric) },
      }),
    });
  }

  async ensureIndex(params: CreateIndexParams): Promise<void> {
    try {
      const current = await this.describeIndex({ indexName: params.indexName });
      if (
        current.dimension !== params.dimension ||
        (params.metric && current.metric !== params.metric)
      ) {
        throw new RagServiceError(
          "RAG_INDEX_MISMATCH",
          `集合 ${params.indexName} 的维度或距离类型不匹配。`,
        );
      }
    } catch (error) {
      if (error instanceof RagServiceError && error.status === 404) {
        await this.createIndex(params);
        return;
      }
      throw error;
    }
  }

  async listAliases(): Promise<QdrantAlias[]> {
    const result = await this.request<{ aliases: QdrantAlias[] }>("/aliases", {
      method: "GET",
    });
    return result.aliases;
  }

  async ensureAliasIndex(params: CreateIndexParams): Promise<void> {
    const existingAlias = (await this.listAliases()).find(
      ({ alias_name }) => alias_name === params.indexName,
    );
    if (existingAlias) {
      await this.ensureIndex({ ...params, indexName: existingAlias.collection_name });
      return;
    }
    try {
      await this.describeIndex({ indexName: params.indexName });
      return;
    } catch (error) {
      if (!(error instanceof RagServiceError) || error.status !== 404) throw error;
    }

    const backingCollection = `${params.indexName}_initial`;
    await this.ensureIndex({ ...params, indexName: backingCollection });
    await this.switchAlias(params.indexName, backingCollection);
  }

  async switchAlias(alias: string, collection: string): Promise<void> {
    const existingAlias = (await this.listAliases()).find(
      ({ alias_name }) => alias_name === alias,
    );
    const actions: Array<Record<string, unknown>> = [];
    if (existingAlias) {
      actions.push({ delete_alias: { alias_name: alias } });
    }
    actions.push({
      create_alias: { alias_name: alias, collection_name: collection },
    });
    await this.request("/collections/aliases", {
      method: "POST",
      body: JSON.stringify({ actions }),
    });
  }

  async listIndexes(): Promise<string[]> {
    const result = await this.request<{ collections: Array<{ name: string }> }>(
      "/collections",
      { method: "GET" },
    );
    return result.collections.map(({ name }) => name);
  }

  async describeIndex(params: DescribeIndexParams): Promise<IndexStats> {
    const result = await this.request<{
      points_count?: number;
      vectors_count?: number;
      config: { params: { vectors: { size: number; distance?: string } } };
    }>(`/collections/${encodeURIComponent(params.indexName)}`, { method: "GET" });
    return {
      dimension: result.config.params.vectors.size,
      count: result.points_count ?? result.vectors_count ?? 0,
      metric: mastraMetric(result.config.params.vectors.distance),
    };
  }

  async deleteIndex(params: DeleteIndexParams): Promise<void> {
    await this.request(`/collections/${encodeURIComponent(params.indexName)}`, {
      method: "DELETE",
    });
  }

  async updateVector(params: UpdateVectorParams<QdrantFilter>): Promise<void> {
    if (params.update.vector) {
      if (!("id" in params) || !params.id) {
        throw new RagServiceError(
          "RAG_UNSUPPORTED_OPERATION",
          "按过滤条件批量修改向量不受支持。",
        );
      }
      await this.request(
        `/collections/${encodeURIComponent(params.indexName)}/points/vectors?wait=true`,
        {
          method: "PUT",
          body: JSON.stringify({
            points: [{ id: params.id, vector: params.update.vector }],
          }),
        },
      );
    }
    if (params.update.metadata) {
      await this.request(
        `/collections/${encodeURIComponent(params.indexName)}/points/payload?wait=true`,
        {
          method: "POST",
          body: JSON.stringify({
            payload: params.update.metadata,
            ...("id" in params && params.id
              ? { points: [params.id] }
              : { filter: qdrantFilter(params.filter) }),
          }),
        },
      );
    }
  }

  async deleteVector(params: DeleteVectorParams): Promise<void> {
    await this.deleteVectors({ indexName: params.indexName, ids: [params.id] });
  }

  async deleteVectors(params: DeleteVectorsParams<QdrantFilter>): Promise<void> {
    if ((params.ids?.length ?? 0) === 0 && !params.filter) {
      throw new RagServiceError("RAG_INVALID_DELETE", "删除操作必须提供 ID 或过滤条件。");
    }
    await this.request(
      `/collections/${encodeURIComponent(params.indexName)}/points/delete?wait=true`,
      {
        method: "POST",
        body: JSON.stringify(
          params.ids ? { points: params.ids } : { filter: qdrantFilter(params.filter) },
        ),
      },
    );
  }
}

export const qdrantVector = new QdrantVectorStore();
