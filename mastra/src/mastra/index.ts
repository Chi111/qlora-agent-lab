import { Mastra } from "@mastra/core/mastra";

import { customerServiceAgent } from "./agents/customer-service.js";
import { storage } from "./storage.js";

export const mastra = new Mastra({
  agents: {
    customerServiceAgent,
  },
  storage,
});
