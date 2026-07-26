export class RagServiceError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly status: number | undefined;

  constructor(
    code: string,
    message: string,
    options: { retryable?: boolean; status?: number; cause?: unknown } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "RagServiceError";
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.status = options.status;
  }
}

export interface JsonRequestOptions {
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

export async function requestJson<T>(
  url: string,
  init: RequestInit,
  options: JsonRequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 15_000);
  timeout.unref?.();

  let response: Response;
  try {
    response = await (options.fetchImpl ?? fetch)(url, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body === undefined ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
        ...init.headers,
      },
      signal: controller.signal,
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "AbortError";
    throw new RagServiceError(
      timedOut ? "RAG_REQUEST_TIMEOUT" : "RAG_SERVICE_UNAVAILABLE",
      timedOut ? "RAG 服务请求超时。" : "RAG 服务暂时不可用。",
      { retryable: true, cause: error },
    );
  } finally {
    clearTimeout(timeout);
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    throw new RagServiceError("RAG_INVALID_RESPONSE", "RAG 服务返回了无效 JSON。", {
      retryable: response.status >= 500,
      status: response.status,
      cause: error,
    });
  }

  if (!response.ok) {
    throw new RagServiceError(
      "RAG_UPSTREAM_ERROR",
      `RAG 上游服务返回 HTTP ${response.status}。`,
      { retryable: response.status >= 500 || response.status === 429, status: response.status },
    );
  }
  return body as T;
}
