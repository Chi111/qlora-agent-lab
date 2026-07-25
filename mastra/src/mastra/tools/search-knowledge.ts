import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const MAX_DOCUMENTS = 100;
const MAX_DOCUMENT_BYTES = 256 * 1024;
const MAX_CONTENT_CHARS = 4_000;

interface KnowledgeDocument {
  document_id: string;
  title: string;
  content: string;
  source: string;
}

export interface KnowledgeSearchResult extends KnowledgeDocument {
  score: number;
}

function tokens(text: string): string[] {
  const raw = text.toLowerCase().match(/[a-z0-9]+|[\u4e00-\u9fff]/g) ?? [];
  const chinese = raw.filter((item) => /^[\u4e00-\u9fff]$/.test(item));
  const bigrams = chinese.slice(0, -1).map((item, index) => `${item}${chinese[index + 1]}`);
  return [...raw, ...bigrams];
}

function termCounts(items: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) {
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  return counts;
}

async function loadDocuments(directory: string): Promise<KnowledgeDocument[]> {
  let names: string[];
  try {
    names = (await readdir(directory))
      .filter((name) => name.toLowerCase().endsWith(".md"))
      .sort()
      .slice(0, MAX_DOCUMENTS);
  } catch {
    return [];
  }

  const documents: KnowledgeDocument[] = [];
  for (const name of names) {
    const filePath = path.join(directory, name);
    const metadata = await stat(filePath);
    if (!metadata.isFile() || metadata.size > MAX_DOCUMENT_BYTES) {
      continue;
    }
    const content = (await readFile(filePath, "utf8")).trim();
    if (!content) {
      continue;
    }
    const firstLine = content.split(/\r?\n/, 1)[0] ?? "";
    const title = firstLine.startsWith("#")
      ? firstLine.replace(/^#+/, "").trim()
      : path.parse(name).name;
    documents.push({
      document_id: path.parse(name).name,
      title,
      content,
      source: name,
    });
  }
  return documents;
}

export async function searchKnowledge(
  query: string,
  directory = process.env.KNOWLEDGE_DIR ?? path.resolve(process.cwd(), "../data/knowledge"),
  topK = 3,
): Promise<KnowledgeSearchResult[]> {
  const documents = await loadDocuments(directory);
  const queryTerms = termCounts(tokens(query));
  if (documents.length === 0 || queryTerms.size === 0) {
    return [];
  }

  const documentCounts = documents.map((document) => termCounts(tokens(document.content)));
  const lengths = documentCounts.map((counts) =>
    [...counts.values()].reduce((total, value) => total + value, 0),
  );
  const averageLength =
    lengths.reduce((total, value) => total + value, 0) / Math.max(lengths.length, 1);
  const documentFrequency = new Map<string, number>();
  for (const counts of documentCounts) {
    for (const term of counts.keys()) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
  }

  return documents
    .map((document, index) => {
      const counts = documentCounts[index] ?? new Map<string, number>();
      if (
        ![...queryTerms.keys()].some(
          (term) => term.length > 1 && (counts.get(term) ?? 0) > 0,
        )
      ) {
        return undefined;
      }
      let score = 0;
      for (const [term, queryFrequency] of queryTerms) {
        const frequency = counts.get(term) ?? 0;
        if (frequency === 0) {
          continue;
        }
        const frequencyInDocuments = documentFrequency.get(term) ?? 0;
        const inverseDocumentFrequency = Math.log(
          1 +
            (documents.length - frequencyInDocuments + 0.5) /
              (frequencyInDocuments + 0.5),
        );
        const denominator =
          frequency +
          1.5 *
            (1 - 0.75 + (0.75 * (lengths[index] ?? 0)) / Math.max(averageLength, 1));
        score +=
          (inverseDocumentFrequency * frequency * 2.5 * queryFrequency) / denominator;
      }
      if (score <= 0) {
        return undefined;
      }
      return {
        ...document,
        content: document.content.slice(0, MAX_CONTENT_CHARS),
        score: Number(score.toFixed(4)),
      };
    })
    .filter((result): result is KnowledgeSearchResult => result !== undefined)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.max(1, Math.min(topK, 5)));
}

const knowledgeResultSchema = z.object({
  document_id: z.string(),
  title: z.string(),
  content: z.string(),
  source: z.string(),
  score: z.number(),
});

export const searchKnowledgeTool = createTool({
  id: "search_knowledge",
  description:
    "查询产品说明、配送规则、退换货和退款政策等非实时知识；不要用于查询具体订单状态。",
  inputSchema: z.object({
    query: z.string().min(1).max(500).describe("需要从本地知识库检索的问题。"),
  }),
  outputSchema: z.object({
    ok: z.boolean(),
    results: z.array(knowledgeResultSchema),
    message: z.string().nullable(),
  }),
  execute: async ({ query }) => {
    const results = await searchKnowledge(query);
    return {
      ok: results.length > 0,
      results,
      message: results.length > 0 ? null : "知识库中没有找到相关内容。",
    };
  },
});
