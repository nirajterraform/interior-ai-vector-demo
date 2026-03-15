import { GoogleGenAI } from "@google/genai";

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

const project = requiredEnv("GOOGLE_CLOUD_PROJECT");
const location = process.env.GOOGLE_CLOUD_LOCATION || "global";

export const genAI = new GoogleGenAI({
  vertexai: true,
  project,
  location,
});

export async function embedQuery(text: string): Promise<number[]> {
  const response = await genAI.models.embedContent({
    model: "gemini-embedding-001",
    contents: [text],
    config: {
      taskType: "RETRIEVAL_QUERY",
      outputDimensionality: 768,
    },
  });

  const values = response.embeddings?.[0]?.values;
  if (!values || !Array.isArray(values) || values.length === 0) {
    throw new Error("Failed to generate query embedding.");
  }

  return values;
}