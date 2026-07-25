import { fileURLToPath, pathToFileURL } from "node:url";

import { LibSQLStore } from "@mastra/libsql";

const configuredUrl = process.env.MASTRA_DB_URL?.trim();
const defaultDatabasePath = fileURLToPath(new URL("../../mastra.db", import.meta.url));

export const storage = new LibSQLStore({
  id: "qlora-customer-service-storage",
  url: configuredUrl || pathToFileURL(defaultDatabasePath).href,
});
