import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
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
import { EMBEDDING_DIMENSION } from "../src/mastra/rag/config.js";
import {
  chunkMarkdown,
  ingestMarkdownDocument,
  ingestMarkdownDirectory,
  rebuildMarkdownDirectory,
} from "../src/mastra/rag/ingestion.js";
import { rerankDocuments } from "../src/mastra/rag/model-service.js";
import { handleInternalKnowledgeSearch } from "../src/mastra/routes/internal-knowledge.js";
import {
  approveExperienceDraft,
  renderRepairExperienceMarkdown,
  saveExperienceDraft,
} from "../src/mastra/workflows/save-repair-experience.js";

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
  it("always reranks vector candidates and returns the reranker order", async () => {
    const vectorStore = {
      query: vi.fn(async () => [
        {
          id: "wrong-first",
          score: 0.95,
          metadata: {
            document_id: "monitor",
            title: "显示器维修",
            source: "monitor.md",
            section: "无信号",
            text: "显示器无信号时检查输入源。",
          },
        },
        {
          id: "correct-second",
          score: 0.72,
          metadata: {
            document_id: "refrigerator",
            title: "冰箱维修",
            source: "refrigerator.md",
            section: "不制冷",
            text: "冰箱不制冷时先检查电源和温控设置。",
          },
        },
      ]),
    };
    const rerank = vi.fn(async (_query: string, documents: string[]) => {
      expect(documents).toEqual([
        "显示器无信号时检查输入源。",
        "冰箱不制冷时先检查电源和温控设置。",
      ]);
      return [
        { index: 1, score: 0.99 },
        { index: 0, score: 0.12 },
      ];
    });

    const response = await searchKnowledge("冰箱为什么不制冷", {
      topK: 2,
      knowledgeScope: "repair",
      vectorStore,
      embed: async () => [Array(EMBEDDING_DIMENSION).fill(0.1)],
      rerank,
      minRerankScore: 0,
    });

    expect(vectorStore.query).toHaveBeenCalledWith(
      expect.objectContaining({ topK: 12, filter: { knowledge_scope: "repair" } }),
    );
    expect(rerank).toHaveBeenCalledOnce();
    expect(response.ok).toBe(true);
    expect(response.reranked).toBe(true);
    expect(response.results.map(({ document_id }) => document_id)).toEqual([
      "refrigerator",
      "monitor",
    ]);
    expect(response.results[0]).toMatchObject({
      rank: 1,
      score: 0.99,
      vector_score: 0.72,
      rerank_score: 0.99,
    });
  });

  it("does not expose vector candidates when reranking fails", async () => {
    const response = await searchKnowledge("冰箱不制冷", {
      vectorStore: {
        query: async () => [
          {
            id: "candidate",
            score: 0.8,
            metadata: { text: "未经重排的候选内容" },
          },
        ],
      },
      embed: async () => [Array(EMBEDDING_DIMENSION).fill(0.1)],
      rerank: async () => {
        throw new Error("reranker offline");
      },
    });

    expect(response).toMatchObject({
      ok: false,
      reranked: false,
      candidate_count: 1,
      results: [],
      error: { code: "RAG_RERANK_UNAVAILABLE", retryable: true },
    });
    expect(response.message).toContain("不会使用未经重排");
  });

  it("returns no answer when every reranked candidate is below threshold", async () => {
    const response = await searchKnowledge("量子引力弦理论", {
      vectorStore: {
        query: async () => [
          {
            id: "irrelevant",
            score: 0.7,
            metadata: { text: "冰箱门封条检查方法" },
          },
        ],
      },
      embed: async () => [Array(EMBEDDING_DIMENSION).fill(0.1)],
      rerank: async () => [{ index: 0, score: 0.05 }],
      minRerankScore: 0.2,
    });

    expect(response).toMatchObject({
      ok: false,
      reranked: true,
      results: [],
    });
    expect(response.message).toContain("相关性阈值");
  });

  it("rejects a truncated reranker response", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({
          results: [{ index: 0, relevance_score: 0.9 }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(
      rerankDocuments("冰箱", ["第一条", "第二条"], 2, {
        baseUrl: "http://reranker.invalid",
        fetchImpl,
      }),
    ).rejects.toThrow("必须返回 2 条");
  });
});

describe("Markdown ingestion", () => {
  it("creates stable chunks with the required retrieval metadata", () => {
    const content = `---
document_id: stable-refrigerator
title: "冰箱不制冷维修"
appliance_type: "refrigerator"
document_version: "2"
updated_at: "2026-07-26"
safety_level: "high"
---

# 冰箱不制冷维修

## 安全红线

不要拆卸压缩机。${"检查电源和温控设置。".repeat(70)}
`;
    const first = chunkMarkdown(content, {
      documentId: "refrigerator-repair",
      source: "repair/refrigerator-repair.md",
      knowledgeScope: "repair",
    });
    const second = chunkMarkdown(content, {
      documentId: "refrigerator-repair",
      source: "repair/refrigerator-repair.md",
      knowledgeScope: "repair",
    });

    expect(first.length).toBeGreaterThan(1);
    expect(first.map(({ id }) => id)).toEqual(second.map(({ id }) => id));
    expect(first[0]?.metadata).toMatchObject({
      document_id: "stable-refrigerator",
      title: "冰箱不制冷维修",
      source: "repair/refrigerator-repair.md",
      knowledge_scope: "repair",
      appliance_type: "refrigerator",
      section: "安全红线",
      chunk_index: 0,
      document_version: "2",
    });
    expect(first[0]?.metadata).toMatchObject({
      document_id: "stable-refrigerator",
      updated_at: "2026-07-26",
      safety_level: "high",
    });
    expect(first[0]?.metadata.content_hash).toMatch(/^[a-f0-9]{64}$/);
    expect(first[0]?.metadata.text).toBe(first[0]?.text);
  });

  it("uses frontmatter document IDs across file renames and rejects duplicates", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-document-id-"));
    const firstPath = path.join(directory, "first-name.md");
    const secondPath = path.join(directory, "renamed.md");
    const content =
      "---\ndocument_id: stable-monitor\nknowledge_scope: repair\n---\n# 显示器\n检查信号线。";
    await writeFile(firstPath, content, "utf8");
    const firstChunks = chunkMarkdown(content, {
      documentId: path.parse(firstPath).name,
      source: "first-name.md",
      knowledgeScope: "repair",
    });
    const renamedChunks = chunkMarkdown(content, {
      documentId: path.parse(secondPath).name,
      source: "renamed.md",
      knowledgeScope: "repair",
    });
    expect(firstChunks[0]?.metadata.document_id).toBe("stable-monitor");
    expect(renamedChunks[0]?.metadata.document_id).toBe("stable-monitor");

    await writeFile(secondPath, content, "utf8");
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      createIndex: vi.fn(async () => undefined),
      upsert: vi.fn(async ({ ids }: { ids: string[] }) => ids),
      describeIndex: vi.fn(async () => ({ dimension: 1024, count: 2 })),
      switchAlias: vi.fn(async () => undefined),
    };
    await expect(
      rebuildMarkdownDirectory(directory, {
        vectorStore,
        embed: async (texts) =>
          texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.2)),
      }),
    ).rejects.toThrow("重复文档 ID");

    await expect(
      ingestMarkdownDirectory(directory, {
        vectorStore,
        embed: async (texts) =>
          texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.2)),
      }),
    ).rejects.toThrow("重复文档 ID");
    expect(vectorStore.deleteVectors).not.toHaveBeenCalled();
  });

  it("embeds, replaces, and upserts a document through the reusable ingestion API", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-ingest-"));
    const filePath = path.join(directory, "monitor.md");
    await writeFile(filePath, "# 显示器维修\n无信号时检查输入源。", "utf8");
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      upsert: vi.fn(async ({ ids }: { ids: string[] }) => ids),
    };

    const result = await ingestMarkdownDocument(filePath, {
      sourceRoot: directory,
      vectorStore,
      embed: async (texts) => texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.2)),
    });

    expect(result).toMatchObject({ documents: 1, chunks: 1 });
    expect(vectorStore.ensureIndex).toHaveBeenCalledWith(
      expect.objectContaining({ dimension: 1024, metric: "cosine" }),
    );
    expect(vectorStore.deleteVectors).toHaveBeenCalledWith(
      expect.objectContaining({
        filter: { document_id: "monitor", knowledge_scope: "repair" },
      }),
    );
    expect(vectorStore.upsert).toHaveBeenCalledOnce();
  });

  it("validates a staging collection before atomically switching the live alias", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-rebuild-"));
    await writeFile(
      path.join(directory, "television.md"),
      "# 彩电维修\n无画面时停止拆机并联系专业维修人员。",
      "utf8",
    );
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      createIndex: vi.fn(async () => undefined),
      upsert: vi.fn(async ({ ids }: { ids: string[] }) => ids),
      describeIndex: vi.fn(async () => ({ dimension: 1024, count: 1 })),
      switchAlias: vi.fn(async () => undefined),
    };

    const result = await rebuildMarkdownDirectory(directory, {
      collection: "repair_knowledge_current",
      vectorStore,
      embed: async (texts) => texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.3)),
    });

    expect(result).toMatchObject({ documents: 1, chunks: 1 });
    expect(vectorStore.createIndex).toHaveBeenCalledWith(
      expect.objectContaining({
        indexName: expect.stringMatching(
          /^repair_knowledge_[a-f0-9]{12}_[a-f0-9]{8}$/,
        ),
      }),
    );
    expect(vectorStore.switchAlias).toHaveBeenCalledWith(
      "repair_knowledge_current",
      expect.stringMatching(/^repair_knowledge_[a-f0-9]{12}_[a-f0-9]{8}$/),
    );
    expect(vectorStore.describeIndex.mock.invocationCallOrder[0]).toBeLessThan(
      vectorStore.switchAlias.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY,
    );
  });

  it("preserves frontmatter scopes and excludes unapproved drafts", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-scopes-"));
    const approvedDirectory = path.join(directory, "experience", "approved");
    const draftDirectory = path.join(directory, "experience", "drafts");
    await mkdir(approvedDirectory, { recursive: true });
    await mkdir(draftDirectory, { recursive: true });
    await writeFile(
      path.join(directory, "monitor.md"),
      "---\nknowledge_scope: repair\n---\n# 显示器\n检查信号线。",
      "utf8",
    );
    await writeFile(
      path.join(approvedDirectory, "lesson.md"),
      "---\nknowledge_scope: experience\n---\n# 已审核经验\n记录向量库经验。",
      "utf8",
    );
    await writeFile(
      path.join(draftDirectory, "rejected.md"),
      "---\nknowledge_scope: experience\n---\n# 未审核草稿\n不得发布。",
      "utf8",
    );
    const upsertedMetadata: Array<Record<string, unknown>> = [];
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      createIndex: vi.fn(async () => undefined),
      upsert: vi.fn(
        async ({
          ids,
          metadata,
        }: {
          ids: string[];
          metadata: Array<Record<string, unknown>>;
        }) => {
          upsertedMetadata.push(...metadata);
          return ids;
        },
      ),
      describeIndex: vi.fn(async () => ({ dimension: 1024, count: 2 })),
      switchAlias: vi.fn(async () => undefined),
    };

    const result = await rebuildMarkdownDirectory(directory, {
      vectorStore,
      embed: async (texts) =>
        texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.2)),
    });

    expect(result.documents).toBe(2);
    expect(
      upsertedMetadata.map((item) => item.knowledge_scope).sort(),
    ).toEqual(["experience", "repair"]);
    expect(
      upsertedMetadata.some((item) => item.document_id === "rejected"),
    ).toBe(false);
  });

  it("changes the staging content hash for frontmatter-only changes", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-frontmatter-"));
    const filePath = path.join(directory, "monitor.md");
    const collectionNames: string[] = [];
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      createIndex: vi.fn(async ({ indexName }: { indexName: string }) => {
        collectionNames.push(indexName);
      }),
      upsert: vi.fn(async ({ ids }: { ids: string[] }) => ids),
      describeIndex: vi.fn(async () => ({ dimension: 1024, count: 1 })),
      switchAlias: vi.fn(async () => undefined),
    };
    const rebuild = async (title: string) => {
      await writeFile(
        filePath,
        `---\ntitle: "${title}"\nknowledge_scope: repair\n---\n# 显示器\n检查信号线。`,
        "utf8",
      );
      await rebuildMarkdownDirectory(directory, {
        vectorStore,
        embed: async (texts) =>
          texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.2)),
      });
    };

    await rebuild("第一版标题");
    await rebuild("第二版标题");

    expect(collectionNames).toHaveLength(2);
    expect(collectionNames[0]?.split("_")[2]).not.toBe(
      collectionNames[1]?.split("_")[2],
    );
  });

  it("keeps the previous alias when staging validation fails", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mastra-rebuild-invalid-"));
    await writeFile(path.join(directory, "monitor.md"), "# 显示器维修\n检查信号线。", "utf8");
    const vectorStore = {
      ensureIndex: vi.fn(async () => undefined),
      deleteVectors: vi.fn(async () => undefined),
      createIndex: vi.fn(async () => undefined),
      upsert: vi.fn(async ({ ids }: { ids: string[] }) => ids),
      describeIndex: vi.fn(async () => ({ dimension: 1024, count: 0 })),
      deleteIndex: vi.fn(async () => undefined),
      switchAlias: vi.fn(async () => undefined),
    };

    await expect(
      rebuildMarkdownDirectory(directory, {
        vectorStore,
        embed: async (texts) =>
          texts.map(() => Array(EMBEDDING_DIMENSION).fill(0.4)),
      }),
    ).rejects.toThrow("未切换正式别名");
    expect(vectorStore.switchAlias).not.toHaveBeenCalled();
    expect(vectorStore.deleteIndex).toHaveBeenCalledOnce();
  });
});

describe("internal knowledge search route", () => {
  it("fails closed when the internal shared key is not configured", async () => {
    const previous = process.env.MASTRA_INTERNAL_SEARCH_KEY;
    delete process.env.MASTRA_INTERNAL_SEARCH_KEY;
    try {
      const response = await handleInternalKnowledgeSearch(
        new Request("http://localhost/internal/knowledge/search", {
          method: "POST",
          body: JSON.stringify({ query: "冰箱不制冷" }),
        }),
      );
      expect(response.status).toBe(503);
      expect(await response.json()).toMatchObject({
        error: { code: "INTERNAL_SEARCH_NOT_CONFIGURED" },
      });
    } finally {
      if (previous) process.env.MASTRA_INTERNAL_SEARCH_KEY = previous;
    }
  });

  it("rejects a wrong internal shared key before parsing the request", async () => {
    const previous = process.env.MASTRA_INTERNAL_SEARCH_KEY;
    process.env.MASTRA_INTERNAL_SEARCH_KEY = "correct-secret";
    try {
      const response = await handleInternalKnowledgeSearch(
        new Request("http://localhost/internal/knowledge/search", {
          method: "POST",
          headers: { "x-internal-api-key": "wrong-secret" },
          body: "not json",
        }),
      );
      expect(response.status).toBe(401);
    } finally {
      if (previous) process.env.MASTRA_INTERNAL_SEARCH_KEY = previous;
      else delete process.env.MASTRA_INTERNAL_SEARCH_KEY;
    }
  });
});

describe("repair experience review", () => {
  const experience = {
    experienceId: "monitor-no-signal-001",
    applianceType: "monitor" as const,
    symptom: "客户 13800138000 反馈显示器无信号",
    diagnosis: "确认输入源选择错误",
    actions: "切换到正确输入源并重新插接信号线",
    result: "画面恢复正常",
    safetyNotes: "禁止拆机，异常时联系专业维修人员",
  };

  it("redacts personal data in generated Markdown", () => {
    const markdown = renderRepairExperienceMarkdown(experience);
    expect(markdown).not.toContain("13800138000");
    expect(markdown).toContain("[手机号已脱敏]");
    expect(markdown).toContain('knowledge_scope: "repair"');
  });

  it("keeps a draft out of the approved directory until explicit approval", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "mastra-experience-"));
    const saved = await saveExperienceDraft(experience, root);
    expect(saved.draftPath).toContain(`${path.sep}drafts${path.sep}`);

    const approvedPath = await approveExperienceDraft(
      experience.experienceId,
      root,
      "reviewer-a",
    );
    expect(approvedPath).toContain(`${path.sep}approved${path.sep}`);
    const approved = await readFile(approvedPath, "utf8");
    expect(approved).toContain("[手机号已脱敏]");
    expect(approved).toContain('review_status: "approved"');
    expect(approved).toContain('approved_by: "reviewer-a"');
  });

  it("rejects reuse of an approved experience ID with different content", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "mastra-experience-conflict-"));
    await saveExperienceDraft(experience, root);
    await approveExperienceDraft(experience.experienceId, root);
    await saveExperienceDraft(
      { ...experience, diagnosis: "另一段不同的诊断过程" },
      root,
    );

    await expect(
      approveExperienceDraft(experience.experienceId, root),
    ).rejects.toThrow("不能用不同内容覆盖");
  });
});
