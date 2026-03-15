type RetryOptions = {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function isRetryableGeminiError(error: unknown): boolean {
  const message = getErrorMessage(error).toLowerCase();

  return (
    message.includes("429") ||
    message.includes("quota") ||
    message.includes("rate limit") ||
    message.includes("rate_limit") ||
    message.includes("resource exhausted") ||
    message.includes("too many requests") ||
    message.includes("exceeded") && message.includes("quota")
  );
}

export async function withGeminiRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const maxRetries = options.maxRetries ?? 8;
  const baseDelayMs = options.baseDelayMs ?? 1200;
  const maxDelayMs = options.maxDelayMs ?? 15000;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (!isRetryableGeminiError(error)) {
        throw error;
      }

      if (attempt === maxRetries) {
        throw error;
      }

      const exponential = baseDelayMs * Math.pow(2, attempt);
      const jitter = Math.floor(Math.random() * 500);
      const delay = Math.min(exponential + jitter, maxDelayMs);

      console.warn(
        `Gemini retryable error detected. attempt=${attempt + 1}/${maxRetries + 1}, waiting ${delay}ms`
      );

      await sleep(delay);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Unknown Gemini retry failure");
}