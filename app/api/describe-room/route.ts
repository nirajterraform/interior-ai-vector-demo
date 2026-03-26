import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { withGeminiRetry } from "@/lib/geminiRetry";

const ai = new GoogleGenAI({
  vertexai: true,
  project: process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || "global",
});

function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}

// Prompt: look at the generated room and describe the furniture
// in a way that will match catalogue products
const DESCRIBE_PROMPT = `
You are an interior design product expert. Look at this room image carefully.

Identify every visible furniture piece and describe it for catalogue matching.

SOFA IDENTIFICATION — MOST IMPORTANT:
Look very carefully at the main seating piece. Determine its SIZE and SHAPE:
- Does it have an L-shape or chaise extension? → "large L shaped sectional sofa"
- Is it wide enough for 3+ people with no L? → "large 3 seater sofa"  
- Is it a small 2-person sofa? → "loveseat sofa"
- ALWAYS include "large" or "sectional" in the description if the sofa is a big piece

CATEGORY RULES for other pieces:
- Table sitting IN FRONT of the sofa (low, centre of room) → ALWAYS "coffee table"
- NEVER use "console table", "sofa table", "entryway table"
- Floor covering → "area rug" + material (jute/wool/cotton)
- Standing light → "floor lamp"
- Table light → "table lamp"
- Wall storage → "bookshelf" or "bookcase"
- Low round seat → "pouf ottoman"

STRICT RULES:
1. Mention sofa FIRST with correct size descriptor
2. Mention coffee table second if visible
3. Mention rug third if visible  
4. Include colour + material for every item
5. Return ONLY a comma-separated list — no explanation, no numbers, no bullets
6. Maximum 6 items — only the most visible and important pieces

Example outputs:
large L shaped sectional sofa beige, rectangular oak wood coffee table, cream jute area rug, solid wood floor lamp, wooden bookcase
large 3 seater beige sofa, low wood coffee table, natural jute area rug, tripod floor lamp
`.trim();

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const imageBase64Input = body?.imageBase64;
    const mimeType = body?.mimeType || "image/jpeg";

    if (!imageBase64Input) {
      return NextResponse.json(
        { ok: false, error: "imageBase64 is required" },
        { status: 400 }
      );
    }

    const imageBase64 = stripDataUrlPrefix(imageBase64Input);

    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
            role: "user",
            parts: [
              { text: DESCRIBE_PROMPT },
              { inlineData: { mimeType, data: imageBase64 } },
            ],
          },
        ],
      })
    );

    const parts = response?.candidates?.[0]?.content?.parts || [];
    const description = parts
      .filter((p: any) => typeof p?.text === "string")
      .map((p: any) => p.text)
      .join("")
      .trim();

    return NextResponse.json({
      ok: true,
      furnitureDescription: description,
    });

  } catch (error) {
    console.error("describe-room error:", error);
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Failed to describe room",
      },
      { status: 500 }
    );
  }
}
