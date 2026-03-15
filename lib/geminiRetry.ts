type GeminiRetryOptions = {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableGeminiError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;

  const e = error as {
    status?: number;
    code?: number;
    message?: string;
    error?: {
      code?: number;
      message?: string;
      status?: string;
    };
  };

  const topLevelMessage = (e.message || "").toLowerCase();
  const nestedMessage = (e.error?.message || "").toLowerCase();
  const combinedMessage = `${topLevelMessage} ${nestedMessage}`;

  const statusCode = e.status ?? e.code ?? e.error?.code;

  if (statusCode === 429) return true;

  if (
    combinedMessage.includes("429") ||
    combinedMessage.includes("quota") ||
    combinedMessage.includes("rate limit") ||
    combinedMessage.includes("resource exhausted") ||
    combinedMessage.includes("too many requests")
  ) {
    return true;
  }

  return false;
}

function computeDelay(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number
): number {
  const exponentialDelay = baseDelayMs * Math.pow(2, attempt - 1);
  const cappedDelay = Math.min(exponentialDelay, maxDelayMs);

  // small jitter to avoid retry bursts
  const jitter = Math.floor(Math.random() * 150);

  return cappedDelay + jitter;
}

export async function withGeminiRetry<T>(
  fn: () => Promise<T>,
  options: GeminiRetryOptions = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelayMs = 400,
    maxDelayMs = 1200,
  } = options;

  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      const retryable = isRetryableGeminiError(error);
      const hasMoreAttempts = attempt <= maxRetries;

      if (!retryable || !hasMoreAttempts) {
        throw error;
      }

      const delay = computeDelay(attempt, baseDelayMs, maxDelayMs);

      console.warn(
        `Gemini retryable error detected. attempt=${attempt}/${maxRetries}, waiting ${delay}ms`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}