import { EMBEDDING_DIMENSION, ragConfig } from "./config.js";
import {
  ingestMarkdownDirectory,
  rebuildMarkdownDirectory,
} from "./ingestion.js";
import { qdrantVector } from "../vectors/qdrant.js";

async function checkServices(): Promise<void> {
  const health = await fetch(`${ragConfig.modelServiceUrl}/health`, {
    signal: AbortSignal.timeout(ragConfig.requestTimeoutMs),
  });
  if (!health.ok) {
    throw new Error(`检索模型服务未就绪：HTTP ${health.status}`);
  }
  const modelStatus = (await health.json()) as {
    models_loaded?: boolean;
    embedding_dimensions?: number;
  };
  if (
    modelStatus.models_loaded !== true ||
    modelStatus.embedding_dimensions !== EMBEDDING_DIMENSION
  ) {
    throw new Error("检索模型尚未加载，或 embedding 维度不是 1024。");
  }

  const index = await qdrantVector.describeIndex({
    indexName: ragConfig.collection,
  });
  if (index.dimension !== EMBEDDING_DIMENSION || index.count < 1) {
    throw new Error("Qdrant 正式集合为空，或向量维度不匹配。");
  }
  console.log(
    JSON.stringify(
      {
        ok: true,
        retrieval_model: modelStatus,
        collection: ragConfig.collection,
        vectors: index.count,
        dimension: index.dimension,
      },
      null,
      2,
    ),
  );
}

async function main(): Promise<void> {
  const args = new Set(process.argv.slice(2));
  if (args.has("--check")) {
    await checkServices();
    return;
  }
  const summary = args.has("--rebuild")
    ? await rebuildMarkdownDirectory()
    : await ingestMarkdownDirectory();
  console.log(JSON.stringify({ ok: true, ...summary }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
