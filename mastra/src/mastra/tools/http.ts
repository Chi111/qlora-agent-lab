export interface ToolError {
  code: string;
  message: string;
  retryable: boolean;
}

export type ToolResult<T> =
  | { ok: true; data: T }
  | { ok: false; status_code?: number; error: ToolError };

export interface FetchOptions {
  baseUrl?: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

const MAX_RESPONSE_BYTES = 1_000_000;

class ResponseTooLargeError extends Error {}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

function isTimeoutError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TimeoutError";
}

async function readResponseText(response: Response): Promise<string> {
  const declaredLength = Number.parseInt(response.headers.get("Content-Length") ?? "", 10);
  if (Number.isFinite(declaredLength) && declaredLength > MAX_RESPONSE_BYTES) {
    throw new ResponseTooLargeError();
  }
  if (!response.body) {
    return "";
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytesRead = 0;
  let text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      bytesRead += value.byteLength;
      if (bytesRead > MAX_RESPONSE_BYTES) {
        await reader.cancel();
        throw new ResponseTooLargeError();
      }
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

export async function requestJson(
  path: string,
  init: RequestInit,
  options: FetchOptions = {},
): Promise<
  | { ok: true; status: number; payload: unknown }
  | { ok: false; status?: number; error: ToolError; payload?: unknown }
> {
  const baseUrl = (options.baseUrl ?? process.env.MOCK_API_URL ?? "http://127.0.0.1:8001")
    .replace(/\/+$/, "");
  const timeoutMs =
    options.timeoutMs ?? Number.parseInt(process.env.TOOL_TIMEOUT_MS ?? "10000", 10);
  const fetchImpl = options.fetchImpl ?? fetch;

  let response: Response;
  try {
    response = await fetchImpl(`${baseUrl}${path}`, {
      ...init,
      signal: AbortSignal.timeout(Number.isFinite(timeoutMs) ? timeoutMs : 10_000),
    });
  } catch (error) {
    const timedOut = isTimeoutError(error);
    return {
      ok: false,
      error: {
        code: timedOut ? "BACKEND_TIMEOUT" : "BACKEND_UNAVAILABLE",
        message: timedOut ? "Mock backend timed out." : errorMessage(error),
        retryable: true,
      },
    };
  }

  let text: string;
  try {
    text = await readResponseText(response);
  } catch (error) {
    const tooLarge = error instanceof ResponseTooLargeError;
    const timedOut = isTimeoutError(error);
    return {
      ok: false,
      status: response.status,
      error: {
        code: tooLarge
          ? "BACKEND_RESPONSE_TOO_LARGE"
          : timedOut
            ? "BACKEND_TIMEOUT"
            : "BACKEND_RESPONSE_READ_FAILED",
        message: tooLarge
          ? "Mock backend response exceeded the safety limit."
          : timedOut
            ? "Mock backend timed out while reading the response."
            : errorMessage(error),
        retryable: !tooLarge,
      },
    };
  }

  let payload: unknown;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    return {
      ok: false,
      status: response.status,
      error: {
        code: "INVALID_BACKEND_RESPONSE",
        message: "Mock backend returned non-JSON content.",
        retryable: false,
      },
    };
  }

  if (!response.ok) {
    const backendError =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof payload.error === "object" &&
      payload.error !== null
        ? payload.error
        : undefined;
    const code =
      backendError && "code" in backendError && typeof backendError.code === "string"
        ? backendError.code
        : "BACKEND_REQUEST_FAILED";
    const message =
      backendError &&
      "message" in backendError &&
      typeof backendError.message === "string"
        ? backendError.message
        : `Mock backend returned HTTP ${response.status}.`;
    const retryable =
      backendError &&
      "retryable" in backendError &&
      typeof backendError.retryable === "boolean"
        ? backendError.retryable
        : response.status >= 500;
    return {
      ok: false,
      status: response.status,
      payload,
      error: { code, message, retryable },
    };
  }

  return { ok: true, status: response.status, payload };
}
