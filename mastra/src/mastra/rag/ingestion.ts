import { createHash, randomUUID } from "node:crypto";
import { readFile, readdir, lstat } from "node:fs/promises";
import path from "node:path";

import { EMBEDDING_DIMENSION, ragConfig } from "./config.js";
import { RagServiceError } from "./http.js";
import { embedTexts } from "./model-service.js";
import { qdrantVector } from "../vectors/qdrant.js";

const MAX_DOCUMENT_BYTES = 512 * 1024;
const MAX_DOCUMENTS = 500;
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 80;
const UPSERT_BATCH_SIZE = 32;
const ALLOWED_KNOWLEDGE_SCOPES = new Set([
  "repair",
  "experience",
  "policy",
  "product",
]);

export interface MarkdownChunk {
  id: string;
  text: string;
  metadata: {
    document_id: string;
    title: string;
    source: string;
    knowledge_scope: string;
    appliance_type: string;
    section: string;
    chunk_index: number;
    document_version: string;
    content_hash: string;
    text: string;
    updated_at?: string;
    safety_level?: string;
    review_status?: string;
    approved_by?: string;
  };
}

export interface IngestionVectorStore {
  ensureIndex(params: {
    indexName: string;
    dimension: number;
    metric: "cosine";
  }): Promise<void>;
  deleteVectors(params: {
    indexName: string;
    filter: Record<string, unknown>;
  }): Promise<void>;
  upsert(params: {
    indexName: string;
    vectors: number[][];
    metadata: Array<Record<string, unknown>>;
    ids: string[];
  }): Promise<string[]>;
}

export interface RebuildVectorStore extends IngestionVectorStore {
  createIndex(params: {
    indexName: string;
    dimension: number;
    metric: "cosine";
  }): Promise<void>;
  describeIndex(params: {
    indexName: string;
  }): Promise<{ dimension: number; count: number }>;
  deleteIndex?(params: { indexName: string }): Promise<void>;
  switchAlias(alias: string, collection: string): Promise<void>;
}

export interface IngestOptions {
  collection?: string;
  knowledgeScope?: string;
  sourceRoot?: string;
  vectorStore?: IngestionVectorStore;
  embed?: (texts: string[]) => Promise<number[][]>;
}

export interface IngestionSummary {
  documents: number;
  chunks: number;
  collection: string;
  document_ids: string[];
}

interface ParsedMarkdown {
  body: string;
  frontmatter: Record<string, string>;
  title: string;
}

function parseMarkdown(content: string, fallbackTitle: string): ParsedMarkdown {
  const normalized = content.replace(/\r\n/g, "\n").trim();
  const frontmatter: Record<string, string> = {};
  let body = normalized;
  if (normalized.startsWith("---\n")) {
    const end = normalized.indexOf("\n---\n", 4);
    if (end !== -1) {
      for (const line of normalized.slice(4, end).split("\n")) {
        const separator = line.indexOf(":");
        if (separator > 0) {
          const key = line.slice(0, separator).trim();
          const value = line
            .slice(separator + 1)
            .trim()
            .replace(/^["']|["']$/g, "");
          if (key) frontmatter[key] = value;
        }
      }
      body = normalized.slice(end + 5).trim();
    }
  }
  const firstHeading = body.match(/^#\s+(.+)$/m)?.[1]?.trim();
  return {
    body,
    frontmatter,
    title: frontmatter.title || firstHeading || fallbackTitle,
  };
}

function documentKnowledgeScope(
  content: string,
  override: string | undefined,
): string {
  const scope =
    override ??
    parseMarkdown(content, "document").frontmatter.knowledge_scope ??
    "repair";
  if (!ALLOWED_KNOWLEDGE_SCOPES.has(scope)) {
    throw new RagServiceError(
      "RAG_INVALID_KNOWLEDGE_SCOPE",
      `不支持的知识范围：${scope}`,
    );
  }
  return scope;
}

function documentIdFor(parsed: ParsedMarkdown, fallback: string): string {
  const documentId = parsed.frontmatter.document_id || fallback;
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(documentId)) {
    throw new RagServiceError(
      "RAG_INVALID_DOCUMENT_ID",
      `文档 ID 不合法：${documentId}`,
    );
  }
  return documentId;
}

function sections(body: string, fallbackTitle: string): Array<{ heading: string; text: string }> {
  const lines = body.split("\n");
  const result: Array<{ heading: string; text: string }> = [];
  let heading = fallbackTitle;
  let buffer: string[] = [];
  const flush = () => {
    const text = buffer.join("\n").trim();
    if (text) result.push({ heading, text });
    buffer = [];
  };
  for (const line of lines) {
    const match = line.match(/^#{1,4}\s+(.+)$/);
    if (match?.[1]) {
      flush();
      heading = match[1].trim();
    } else {
      buffer.push(line);
    }
  }
  flush();
  return result;
}

function splitText(text: string): string[] {
  if (text.length <= CHUNK_SIZE) return [text];
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + CHUNK_SIZE, text.length);
    if (end < text.length) {
      const window = text.slice(start, end);
      const boundary = Math.max(
        window.lastIndexOf("\n"),
        window.lastIndexOf("。"),
        window.lastIndexOf("；"),
      );
      if (boundary >= Math.floor(CHUNK_SIZE * 0.6)) {
        end = start + boundary + 1;
      }
    }
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end >= text.length) break;
    start = Math.max(end - CHUNK_OVERLAP, start + 1);
  }
  return chunks;
}

function stableUuid(seed: string): string {
  const hash = createHash("sha256").update(seed).digest("hex").slice(0, 32).split("");
  hash[12] = "4";
  hash[16] = ((Number.parseInt(hash[16] ?? "0", 16) & 0x3) | 0x8).toString(16);
  const value = hash.join("");
  return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20)}`;
}

export function chunkMarkdown(
  content: string,
  options: {
    documentId: string;
    source: string;
    knowledgeScope: string;
    applianceType?: string;
  },
): MarkdownChunk[] {
  const parsed = parseMarkdown(content, options.documentId);
  const documentId = documentIdFor(parsed, options.documentId);
  const contentHash = createHash("sha256").update(parsed.body).digest("hex");
  const version = parsed.frontmatter.document_version || contentHash.slice(0, 12);
  const applianceType =
    options.applianceType || parsed.frontmatter.appliance_type || "general";
  const chunks: MarkdownChunk[] = [];
  for (const section of sections(parsed.body, parsed.title)) {
    for (const text of splitText(section.text)) {
      const chunkIndex = chunks.length;
      const chunkText = `${parsed.title}\n${section.heading}\n${text}`.trim();
      chunks.push({
        id: stableUuid(`${documentId}:${version}:${chunkIndex}:${chunkText}`),
        text: chunkText,
        metadata: {
          document_id: documentId,
          title: parsed.title,
          source: options.source,
          knowledge_scope: options.knowledgeScope,
          appliance_type: applianceType,
          section: section.heading,
          chunk_index: chunkIndex,
          document_version: version,
          content_hash: contentHash,
          text: chunkText,
          ...(parsed.frontmatter.updated_at
            ? { updated_at: parsed.frontmatter.updated_at }
            : {}),
          ...(parsed.frontmatter.safety_level
            ? { safety_level: parsed.frontmatter.safety_level }
            : {}),
          ...(parsed.frontmatter.review_status
            ? { review_status: parsed.frontmatter.review_status }
            : {}),
          ...(parsed.frontmatter.approved_by
            ? { approved_by: parsed.frontmatter.approved_by }
            : {}),
        },
      });
    }
  }
  return chunks;
}

async function assertReadableMarkdown(filePath: string): Promise<void> {
  const metadata = await lstat(filePath);
  if (!metadata.isFile() || metadata.size === 0 || metadata.size > MAX_DOCUMENT_BYTES) {
    throw new RagServiceError(
      "RAG_INVALID_DOCUMENT",
      `Markdown 文件必须非空且不超过 ${MAX_DOCUMENT_BYTES} 字节。`,
    );
  }
}

async function embedChunks(
  chunks: MarkdownChunk[],
  embed: (texts: string[]) => Promise<number[][]>,
): Promise<number[][]> {
  const embeddings: number[][] = [];
  for (let start = 0; start < chunks.length; start += UPSERT_BATCH_SIZE) {
    embeddings.push(
      ...(await embed(
        chunks.slice(start, start + UPSERT_BATCH_SIZE).map(({ text }) => text),
      )),
    );
  }
  if (
    embeddings.length !== chunks.length ||
    embeddings.some((embedding) => embedding.length !== EMBEDDING_DIMENSION)
  ) {
    throw new RagServiceError(
      "RAG_EMBEDDING_INVALID",
      `入库向量必须为 ${EMBEDDING_DIMENSION} 维且与切片一一对应。`,
    );
  }
  return embeddings;
}

export async function ingestMarkdownDocument(
  filePath: string,
  options: IngestOptions = {},
): Promise<IngestionSummary> {
  await assertReadableMarkdown(filePath);
  const root = path.resolve(options.sourceRoot ?? path.dirname(filePath));
  const resolvedFile = path.resolve(filePath);
  const relative = path.relative(root, resolvedFile);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new RagServiceError("RAG_INVALID_DOCUMENT_PATH", "Markdown 文件不在允许的目录内。");
  }
  if (relative.split(path.sep).some((segment) => segment.toLowerCase() === "drafts")) {
    throw new RagServiceError(
      "RAG_UNAPPROVED_DOCUMENT",
      "草稿目录中的文档不能写入正式向量库。",
    );
  }

  const content = await readFile(resolvedFile, "utf8");
  const knowledgeScope = documentKnowledgeScope(content, options.knowledgeScope);
  const chunks = chunkMarkdown(content, {
    documentId: path.parse(resolvedFile).name,
    source: relative.replaceAll(path.sep, "/"),
    knowledgeScope,
  });
  if (chunks.length === 0) {
    throw new RagServiceError("RAG_INVALID_DOCUMENT", "Markdown 文件没有可入库内容。");
  }
  const documentId = chunks[0]?.metadata.document_id;
  if (!documentId) {
    throw new RagServiceError("RAG_INVALID_DOCUMENT", "文档缺少稳定 ID。");
  }

  const collection = options.collection ?? ragConfig.collection;
  const vectorStore = options.vectorStore ?? qdrantVector;
  if ("ensureAliasIndex" in vectorStore) {
    await (
      vectorStore as IngestionVectorStore & {
        ensureAliasIndex: IngestionVectorStore["ensureIndex"];
      }
    ).ensureAliasIndex({
      indexName: collection,
      dimension: EMBEDDING_DIMENSION,
      metric: "cosine",
    });
  } else {
    await vectorStore.ensureIndex({
      indexName: collection,
      dimension: EMBEDDING_DIMENSION,
      metric: "cosine",
    });
  }
  const embeddings = await embedChunks(chunks, options.embed ?? embedTexts);

  await vectorStore.deleteVectors({
    indexName: collection,
    filter: { document_id: documentId, knowledge_scope: knowledgeScope },
  });
  for (let start = 0; start < chunks.length; start += UPSERT_BATCH_SIZE) {
    const batch = chunks.slice(start, start + UPSERT_BATCH_SIZE);
    await vectorStore.upsert({
      indexName: collection,
      vectors: embeddings.slice(start, start + UPSERT_BATCH_SIZE),
      metadata: batch.map(({ metadata }) => metadata),
      ids: batch.map(({ id }) => id),
    });
  }
  return {
    documents: 1,
    chunks: chunks.length,
    collection,
    document_ids: [documentId],
  };
}

async function markdownFiles(directory: string): Promise<string[]> {
  const names = (await readdir(directory, { withFileTypes: true })).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  const files: string[] = [];
  for (const entry of names) {
    if (files.length >= MAX_DOCUMENTS) break;
    const filePath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.toLowerCase() === "drafts") continue;
      files.push(...(await markdownFiles(filePath)));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      files.push(filePath);
    }
  }
  return files.slice(0, MAX_DOCUMENTS);
}

async function assertUniqueDocumentIds(files: string[]): Promise<void> {
  const documentIds = new Set<string>();
  for (const filePath of files) {
    await assertReadableMarkdown(filePath);
    const parsed = parseMarkdown(
      await readFile(filePath, "utf8"),
      path.parse(filePath).name,
    );
    const documentId = documentIdFor(parsed, path.parse(filePath).name);
    if (documentIds.has(documentId)) {
      throw new RagServiceError(
        "RAG_DUPLICATE_DOCUMENT_ID",
        `发现重复文档 ID：${documentId}`,
      );
    }
    documentIds.add(documentId);
  }
}

export async function ingestMarkdownDirectory(
  directory = ragConfig.knowledgeDirectory,
  options: IngestOptions = {},
): Promise<IngestionSummary> {
  const root = path.resolve(directory);
  const files = await markdownFiles(root);
  await assertUniqueDocumentIds(files);
  const summary: IngestionSummary = {
    documents: 0,
    chunks: 0,
    collection: options.collection ?? ragConfig.collection,
    document_ids: [],
  };
  for (const filePath of files) {
    const result = await ingestMarkdownDocument(filePath, {
      ...options,
      sourceRoot: root,
    });
    summary.documents += result.documents;
    summary.chunks += result.chunks;
    summary.document_ids.push(...result.document_ids);
  }
  return summary;
}

export async function rebuildMarkdownDirectory(
  directory = ragConfig.knowledgeDirectory,
  options: Omit<IngestOptions, "vectorStore"> & {
    vectorStore?: RebuildVectorStore;
  } = {},
): Promise<IngestionSummary> {
  const root = path.resolve(directory);
  const files = await markdownFiles(root);
  const allChunks: MarkdownChunk[] = [];
  const documentIds = new Set<string>();
  for (const filePath of files) {
    await assertReadableMarkdown(filePath);
    const content = await readFile(filePath, "utf8");
    const relative = path.relative(root, filePath).replaceAll(path.sep, "/");
    const knowledgeScope = documentKnowledgeScope(content, options.knowledgeScope);
    const chunks = chunkMarkdown(content, {
      documentId: path.parse(filePath).name,
      source: relative,
      knowledgeScope,
    });
    const documentId = chunks[0]?.metadata.document_id;
    if (!documentId) {
      throw new RagServiceError("RAG_INVALID_DOCUMENT", `${relative} 没有可入库内容。`);
    }
    if (documentIds.has(documentId)) {
      throw new RagServiceError(
        "RAG_DUPLICATE_DOCUMENT_ID",
        `发现重复文档 ID：${documentId}`,
      );
    }
    documentIds.add(documentId);
    allChunks.push(...chunks);
  }
  if (allChunks.length === 0) {
    throw new RagServiceError("RAG_INVALID_DOCUMENT", "知识目录中没有可入库内容。");
  }

  const alias = options.collection ?? ragConfig.collection;
  const versionHash = createHash("sha256")
    .update(
      JSON.stringify(
        allChunks.map(({ id, metadata }) => ({
          id,
          metadata,
        })),
      ),
    )
    .digest("hex")
    .slice(0, 12);
  const stagingCollection =
    `repair_knowledge_${versionHash}_${randomUUID().replaceAll("-", "").slice(0, 8)}`;
  const vectorStore = options.vectorStore ?? qdrantVector;
  await vectorStore.createIndex({
    indexName: stagingCollection,
    dimension: EMBEDDING_DIMENSION,
    metric: "cosine",
  });
  try {
    const embeddings = await embedChunks(allChunks, options.embed ?? embedTexts);
    for (let start = 0; start < allChunks.length; start += UPSERT_BATCH_SIZE) {
      const batch = allChunks.slice(start, start + UPSERT_BATCH_SIZE);
      await vectorStore.upsert({
        indexName: stagingCollection,
        vectors: embeddings.slice(start, start + UPSERT_BATCH_SIZE),
        metadata: batch.map(({ metadata }) => metadata),
        ids: batch.map(({ id }) => id),
      });
    }
    const stats = await vectorStore.describeIndex({ indexName: stagingCollection });
    if (stats.dimension !== EMBEDDING_DIMENSION || stats.count !== allChunks.length) {
      throw new RagServiceError(
        "RAG_INDEX_VALIDATION_FAILED",
        "暂存集合数量或维度校验失败，未切换正式别名。",
      );
    }
  } catch (error) {
    try {
      await vectorStore.deleteIndex?.({ indexName: stagingCollection });
    } catch {
      // Preserve the original ingestion error; orphan cleanup is best effort.
    }
    throw error;
  }
  await vectorStore.switchAlias(alias, stagingCollection);
  return {
    documents: files.length,
    chunks: allChunks.length,
    collection: alias,
    document_ids: [...documentIds],
  };
}
