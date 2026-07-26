import { describe, expect, it } from "vitest";

import { toolResultHasError } from "./mastra-client";

describe("tool result status", () => {
  it("treats transport and business failures as errors", () => {
    expect(toolResultHasError({ isError: true, result: { ok: true } })).toBe(true);
    expect(toolResultHasError({ result: { ok: false } })).toBe(true);
    expect(
      toolResultHasError({
        result: { error: { code: "RAG_RERANK_UNAVAILABLE" } },
      }),
    ).toBe(true);
  });

  it("keeps successful tool results complete", () => {
    expect(toolResultHasError({ result: { ok: true } })).toBe(false);
    expect(toolResultHasError({ result: { results: [] } })).toBe(false);
  });
});
