import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

const localProvider = createOpenAICompatible({
  name: "local-qlora",
  baseURL: process.env.LOCAL_LLM_BASE_URL ?? "http://127.0.0.1:8000/v1",
  apiKey: process.env.LOCAL_LLM_API_KEY ?? "local-not-secret",
  includeUsage: true,
});

export const localCustomerServiceModel = localProvider(
  process.env.LOCAL_LLM_MODEL ?? "local-qlora",
);
