import { Mastra } from "@mastra/core/mastra";

import { customerServiceAgent } from "./agents/customer-service.js";
import { internalKnowledgeSearchRoute } from "./routes/internal-knowledge.js";
import { storage } from "./storage.js";
import { qdrantVector } from "./vectors/qdrant.js";
import { saveRepairExperienceWorkflow } from "./workflows/save-repair-experience.js";

export const mastra = new Mastra({
  agents: {
    customerServiceAgent,
  },
  vectors: {
    repairKnowledge: qdrantVector,
  },
  workflows: {
    saveRepairExperienceWorkflow,
  },
  storage,
  server: {
    host: "127.0.0.1",
    apiRoutes: [internalKnowledgeSearchRoute],
    cors: {
      origin: process.env.MASTRA_WEB_ORIGIN ?? "http://127.0.0.1:5173",
      allowMethods: ["GET", "POST", "OPTIONS"],
      allowHeaders: [
        "Content-Type",
        "Authorization",
        "x-mastra-client-type",
        "x-internal-api-key",
      ],
      credentials: false,
    },
  },
});
