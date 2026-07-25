import { Memory } from "@mastra/memory";

import { storage } from "./storage.js";

export const customerServiceMemory = new Memory({
  storage,
  options: {
    lastMessages: 20,
  },
});
