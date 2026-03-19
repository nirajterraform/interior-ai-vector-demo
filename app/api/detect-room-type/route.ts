import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { withGeminiRetry } from "@/lib/geminiRetry";

const ai = new GoogleGenAI({
  vertexai: true,
  project: process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || "global",
});

type RoomType =
  | "living_room"
  | "bedroom"
  | "dining_room"
  | "kitchen"
  | "office";

const VALID_ROOM_TYPES = new Set<RoomType>([
  "living_room",
  "bedroom",
  "dining_room",
  "kitchen",
  "office",
]);

function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}

const DETECTION_PROMPT = `
You are an interior room type classifier.

Look at this room photograph and identify what type of room it is.

Choose exactly one of these room types:
- living_room: a lounge, sitting room, family room, or living area with sofas/chairs
- bedroom: a room for sleeping with a bed as the main feature
- dining_room: a room primarily for eating with a dining table and chairs
- kitchen: a room for cooking with countertops, appliances, and cabinets
- office: a home office or study with a desk as the main feature

Rules:
1. Return ONLY a JSON object — no explanation, no markdown, no extra text
2. If you are confident, set confidence to "high"
3. If the room is ambiguous or unclear, set confidence to "low" and still pick the closest match
4. If the image does not show an indoor room at all, set room_type to null

Return this exact JSON shape:
{
  "room_type": "living_room",
  "confidence": "high",
  "reason": "one short sentence"
}
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
              { text: DETECTION_PROMPT },
              { inlineData: { mimeType, data: imageBase64 } },
            ],
          },
        ],
      })
    );

    const parts = response?.candidates?.[0]?.content?.parts || [];
    const text = parts
      .filter((p: any) => typeof p?.text === "string")
      .map((p: any) => p.text)
      .join("")
      .trim();

    // Parse the JSON response
    let parsed: { room_type: string | null; confidence: string; reason: string };
    try {
      const cleaned = text.replace(/```json|```/g, "").trim();
      const match = cleaned.match(/\{[\s\S]*\}/);
      parsed = JSON.parse(match ? match[0] : cleaned);
    } catch {
      return NextResponse.json(
        { ok: false, error: "Failed to parse room detection response" },
        { status: 500 }
      );
    }

    const detectedType = parsed?.room_type;

    // Validate it's one of our known room types
    if (!detectedType || !VALID_ROOM_TYPES.has(detectedType as RoomType)) {
      return NextResponse.json({
        ok: true,
        roomType: null,
        confidence: "low",
        reason: parsed?.reason || "Could not identify a supported room type",
      });
    }

    return NextResponse.json({
      ok: true,
      roomType: detectedType as RoomType,
      confidence: parsed?.confidence || "high",
      reason: parsed?.reason || "",
    });
  } catch (error) {
    console.error("detect-room-type error:", error);
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Failed to detect room type",
      },
      { status: 500 }
    );
  }
}
