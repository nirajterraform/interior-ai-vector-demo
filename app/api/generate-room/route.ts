import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI, Modality } from "@google/genai";
import sharp from "sharp";
import { withGeminiRetry } from "@/lib/geminiRetry";
 
const ai = new GoogleGenAI({
  vertexai: true,
  project: process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || "global",
});
 
type ShortlistItem = {
  bucket: string;
  product_handle: string;
  title: string;
  image_url: string | null;
  category?: string | null;
  subcategory?: string | null;
  normalized_category?: string | null;
  min_price?: number | null;
  max_price?: number | null;
  similarity?: number;
  pinned?: boolean;
  source?: "catalog" | "innovative";
};
 
type GenerateRoomBody = {
  roomType?: string;
  theme?: string;
  cleanedRoomBase64?: string;
  originalRoomBase64?: string;
  mimeType?: string;
  shortlist?: ShortlistItem[];
  editMode?: boolean;
  baseGeneratedImage?: string | null;
};
 
type Level = "none" | "minor" | "moderate" | "significant";
type Quality = "poor" | "acceptable" | "good" | "excellent";
 
type FurnishedRoomValidation = {
  room_type_correct: boolean;
  theme_match_level: Quality;
  geometry_preserved: boolean;
  windows_preserved: boolean;
  doors_preserved: boolean;
  curtains_preserved: boolean;
  floor_preserved: boolean;
  ceiling_preserved: boolean;
  structural_drift_level: Level;
  catalog_alignment_level: Quality;
  hallucinated_objects_level: Level;
  artifacts_level: Level;
  overall_quality: Quality;
  pass: boolean;
  reason: string;
};
 
type InnovativeProductDetection = {
  title: string;
  category: string;
  bbox: [number, number, number, number];
};
 
type RecommendationRanking = {
  catalog_product_handles: string[];
  innovative_products: InnovativeProductDetection[];
};
 
function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}
 
function extractTextFromResponse(response: any): string {
  const parts = response?.candidates?.[0]?.content?.parts || [];
  let text = "";
 
  for (const part of parts) {
    if (typeof part?.text === "string") {
      text += part.text;
    }
  }
 
  return text.trim();
}
 
function extractImageFromResponse(
  response: any
): { data: string; mimeType: string } | null {
  const parts = response?.candidates?.[0]?.content?.parts || [];
 
  for (const part of parts) {
    if (part?.inlineData?.data) {
      return {
        data: part.inlineData.data,
        mimeType: part.inlineData.mimeType || "image/png",
      };
    }
  }
 
  return null;
}
 
function safeParseJson<T>(text: string): T {
  const cleaned = text.trim();
 
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]) as T;
    }
    throw new Error(
      `No valid JSON object found in model output: ${cleaned.slice(0, 1200)}`
    );
  }
}
 
async function urlToInlineData(url: string) {
  const normalizedUrl = decodeURI(url.trim());
 
  const tryFetch = async () => {
    return await fetch(normalizedUrl, {
      method: "GET",
      redirect: "follow",
      cache: "no-store",
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        Accept:
          "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        Referer: "https://www.shopify.com/",
      },
    });
  };
 
  let res = await tryFetch();
 
  if (!res.ok) {
    await new Promise((r) => setTimeout(r, 300));
    res = await tryFetch();
  }
 
  if (!res.ok) {
    throw new Error(
      `Failed to fetch catalogue image: ${normalizedUrl} (status ${res.status})`
    );
  }
 
  const contentType = res.headers.get("content-type") || "image/jpeg";
  const arrayBuffer = await res.arrayBuffer();
  const base64 = Buffer.from(arrayBuffer).toString("base64");
 
  return {
    inlineData: {
      mimeType: contentType,
      data: base64,
    },
  };
}
 
function shortlistSummary(shortlist: ShortlistItem[]): string {
  return shortlist
    .map((item, idx) => {
      const parts = [
        `${idx + 1}.`,
        `bucket=${item.bucket}`,
        `product_handle=${item.product_handle}`,
        `title=${item.title}`,
        item.category ? `category=${item.category}` : "",
        item.normalized_category
          ? `normalized_category=${item.normalized_category}`
          : "",
      ].filter(Boolean);
      return parts.join("; ");
    })
    .join("\n");
}
 
function buildGeneratePrompt(
  roomType: string,
  theme: string,
  shortlist: ShortlistItem[],
  attemptNumber: number,
  editMode: boolean
) {
  const editText = editMode
    ? `
This is an edit request on an already generated room.
Apply the user's requested changes while preserving the rest of the room as much as possible.
Only change what is necessary.
`
    : "";
 
  const retryText =
    attemptNumber > 1
      ? `
This is retry attempt ${attemptNumber}.
Be more strict about:
- using only catalog-aligned products
- preserving architecture exactly
- applying the requested edit/color/material/style without changing unrelated parts
`
      : "";
 
  return `
You are an expert interior room redesign model.
 
Task:
Redesign this ${roomType.replaceAll(
    "_",
    " "
  )} using only the provided catalogue reference products and the user's design intent.
 
User design intent:
${theme}
 
Catalogue references:
${shortlistSummary(shortlist)}
 
Strict rules:
1. Preserve the exact room geometry.
2. Preserve walls, windows, curtains, doors, ceiling, floor, skirting, trims, camera angle, perspective, and room proportions.
3. Use the cleaned room image as the structural base.
4. Use only the provided catalogue reference images as the furniture and decor inspiration.
5. Add or replace only movable furniture and decor elements.
6. Do NOT alter the architecture.
7. Do NOT add extra windows, doors, walls, or structural features.
8. Keep the output photorealistic and consistent with the user's requested style.
9. Keep the room type correct for the selected room type.
10. Keep furniture and decor visually aligned with the shortlisted catalogue references.
11. Do NOT invent unrelated furniture families outside the shortlist unless absolutely necessary for the user request.
12. If the user requests changes such as blue sofa, velvet sofa, more luxury, warmer tone, or similar edits, apply those style/color/material changes to the shortlisted furniture style rather than inventing unrelated products.
 
${editText}
${retryText}
 
Return one final redesigned room image only.
`.trim();
}
 
function buildValidationPrompt(
  roomType: string,
  theme: string,
  shortlist: ShortlistItem[]
) {
  return `
You are a strict furnished-room validator.
 
You will compare:
1. the original room image
2. the cleaned room image
3. the final furnished room image
4. the selected room type
5. the user design prompt
6. the shortlisted catalog products summary
 
Selected room type:
${roomType}
 
User design prompt:
${theme}
 
Shortlisted catalog products:
${shortlistSummary(shortlist)}
 
Evaluate whether the final furnished room is acceptable.
 
Checks:
1. Is the final room still the same room structurally?
2. Are walls, windows, curtains, doors, ceiling, and floor preserved?
3. Does the final room still match the selected room type?
4. Does the design match the requested style/theme?
5. Does the furniture and decor look consistent with the shortlisted catalog products?
6. Are there obvious hallucinated objects unrelated to the shortlist?
7. Are there visible visual artifacts or broken generated objects?
 
Return strict JSON only:
{
  "room_type_correct": true,
  "theme_match_level": "good",
  "geometry_preserved": true,
  "windows_preserved": true,
  "doors_preserved": true,
  "curtains_preserved": true,
  "floor_preserved": true,
  "ceiling_preserved": true,
  "structural_drift_level": "minor",
  "catalog_alignment_level": "good",
  "hallucinated_objects_level": "minor",
  "artifacts_level": "minor",
  "overall_quality": "good",
  "pass": true,
  "reason": "Short explanation"
}
`.trim();
}
 
function buildRecommendationRankingPrompt(shortlist: ShortlistItem[]) {
  return `
You are a visual product matching expert.
 
You will be shown:
- Image 1: the generated furnished room
- Images 2 onwards: catalogue product reference images, each labelled with its product_handle
 
Your tasks:
1. For each catalogue product image shown, decide if that exact product (or a very close visual match) appears in the generated room. Only match on strong visual similarity — same shape, colour, and style. Do NOT match based on category alone.
2. Return catalog_product_handles for products that are clearly visible in the room, ordered most-to-least prominent.
3. For any significant furniture/decor item visible in the room that does NOT match any catalogue product, list it under innovative_products with a bounding box on a 0–1000 coordinate scale [x1, y1, x2, y2].
 
Catalogue product handles (in the same order as the reference images you will see):
${shortlistSummary(shortlist)}
 
Return strict JSON only — no markdown:
{
  "catalog_product_handles": ["handle1", "handle2"],
  "innovative_products": [
    {
      "title": "Round Glass Coffee Table",
      "category": "tables",
      "bbox": [300, 500, 700, 800]
    }
  ]
}
`.trim();
}
 
function levelScore(level: Level): number {
  switch (level) {
    case "none":
      return 3;
    case "minor":
      return 2;
    case "moderate":
      return 1;
    case "significant":
      return 0;
    default:
      return 0;
  }
}
 
function qualityScore(q: Quality): number {
  switch (q) {
    case "excellent":
      return 4;
    case "good":
      return 3;
    case "acceptable":
      return 2;
    case "poor":
      return 0;
    default:
      return 0;
  }
}
 
function computeValidationScore(v: FurnishedRoomValidation): number {
  let score = 0;
 
  if (v.room_type_correct) score += 4;
  if (v.geometry_preserved) score += 4;
  if (v.windows_preserved) score += 3;
  if (v.doors_preserved) score += 1;
  if (v.curtains_preserved) score += 2;
  if (v.floor_preserved) score += 3;
  if (v.ceiling_preserved) score += 2;
 
  score += qualityScore(v.theme_match_level) * 2;
  score += qualityScore(v.catalog_alignment_level) * 2;
  score += levelScore(v.structural_drift_level) * 3;
  score += levelScore(v.hallucinated_objects_level) * 3;
  score += levelScore(v.artifacts_level) * 2;
  score += qualityScore(v.overall_quality) * 2;
 
  if (v.pass) score += 2;
 
  return score;
}
 
async function buildReferenceParts(shortlist: ShortlistItem[]) {
  const resolvedReferenceParts = await Promise.all(
    shortlist.slice(0, 12).map(async (item) => {
      try {
        const part = await urlToInlineData(item.image_url as string);
        return { ok: true as const, part, item };
      } catch (err: any) {
        // Silently drop 404s — expected for deleted Shopify products
        const status = String(err?.message ?? "").match(/status (\d+)/)?.[1];
        if (status !== "404") {
          console.warn("Skipping catalogue image:", item.image_url, status ?? String(err));
        }
        return { ok: false as const, part: null, item };
      }
    })
  );
 
  // Only items whose images actually fetched — 404s excluded from everything
  const fetched = resolvedReferenceParts.filter((x) => x.ok && x.part !== null);
 
  const referenceParts = fetched.map((x) => x.part);
  const validShortlist = fetched.map((x) => x.item);
 
  return {
    referenceParts,
    validShortlist,
    // Raw inlineData parts kept for reuse in detectPinnedProducts visual matching
    catalogueImageParts: fetched.map((x) => x.part) as { inlineData: { mimeType: string; data: string } }[],
  };
}
 
async function generateRoomAttempt(
  roomType: string,
  theme: string,
  cleanedRoomBase64: string,
  cleanedMimeType: string,
  shortlist: ShortlistItem[],
  attemptNumber: number,
  editMode: boolean,
  baseGeneratedImage?: string | null
) {
  const { referenceParts, validShortlist, catalogueImageParts } = await buildReferenceParts(shortlist);
 
  if (!referenceParts.length) {
    throw new Error("None of the catalogue images could be fetched from Shopify CDN.");
  }
 
  const prompt = buildGeneratePrompt(roomType, theme, validShortlist, attemptNumber, editMode);
 
  const parts: any[] = [{ text: prompt }, ...referenceParts];
 
  if (editMode && baseGeneratedImage) {
    parts.push({ text: "Base image to edit:" });
    parts.push({
      inlineData: {
        mimeType: cleanedMimeType,
        data: stripDataUrlPrefix(baseGeneratedImage),
      },
    });
  } else {
    parts.push({
      inlineData: {
        mimeType: cleanedMimeType,
        data: cleanedRoomBase64,
      },
    });
  }
 
  const response = await withGeminiRetry(() =>
    ai.models.generateContent({
      model: "gemini-2.5-flash-image",
      contents: [
        {
          role: "user",
          parts,
        },
      ],
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      },
    })
  );
 
  const image = extractImageFromResponse(response);
  const rawText = extractTextFromResponse(response) || null;
 
  if (!image) {
    throw new Error(
      `Gemini did not return a redesigned room image.${rawText ? ` Model text: ${rawText}` : ""}`
    );
  }
 
  return {
    imageBase64: image.data,
    mimeType: image.mimeType,
    rawText,
    usedReferenceCount: validShortlist.length,
    validShortlist,
    // Pass through the fetched catalogue images so detectPinnedProducts
    // can use them for visual matching without re-fetching from CDN
    catalogueImageParts,
  };
}
 
async function validateGeneratedRoom(
  originalRoomBase64: string,
  originalMimeType: string,
  cleanedRoomBase64: string,
  cleanedMimeType: string,
  generatedImageBase64: string,
  generatedMimeType: string,
  roomType: string,
  theme: string,
  shortlist: ShortlistItem[]
) {
  const response = await withGeminiRetry(() =>
    ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        {
          role: "user",
          parts: [
            { text: buildValidationPrompt(roomType, theme, shortlist.slice(0, 12)) },
            { text: "Image 1: original room image." },
            {
              inlineData: {
                mimeType: originalMimeType,
                data: originalRoomBase64,
              },
            },
            { text: "Image 2: cleaned room image." },
            {
              inlineData: {
                mimeType: cleanedMimeType,
                data: cleanedRoomBase64,
              },
            },
            { text: "Image 3: final furnished generated room image." },
            {
              inlineData: {
                mimeType: generatedMimeType,
                data: generatedImageBase64,
              },
            },
          ],
        },
      ],
    })
  );
 
  const text = extractTextFromResponse(response);
  return safeParseJson<FurnishedRoomValidation>(text);
}
 
function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}
 
async function cropInnovativeSnapshots(
  generatedImageBase64: string,
  generatedMimeType: string,
  innovativeProducts: InnovativeProductDetection[]
): Promise<ShortlistItem[]> {
  const imageBuffer = Buffer.from(generatedImageBase64, "base64");
  const image = sharp(imageBuffer);
  const metadata = await image.metadata();
 
  const width = metadata.width ?? 0;
  const height = metadata.height ?? 0;
 
  if (!width || !height) {
    return innovativeProducts.map((item, idx) => ({
      bucket: "innovative",
      product_handle: `innovative-${idx}-${item.title.toLowerCase().replace(/\s+/g, "-")}`,
      title: item.title,
      category: item.category,
      subcategory: item.category,
      normalized_category: item.category,
      image_url: null,
      min_price: null,
      max_price: null,
      similarity: 1,
      source: "innovative",
      pinned: true,
    }));
  }
 
  const results: ShortlistItem[] = [];
 
  for (let idx = 0; idx < innovativeProducts.length; idx++) {
    const item = innovativeProducts[idx];
    const [nx1, ny1, nx2, ny2] = item.bbox || [200, 200, 800, 800];
 
    const rawLeft = Math.floor((clamp(nx1, 0, 1000) / 1000) * width);
    const rawTop = Math.floor((clamp(ny1, 0, 1000) / 1000) * height);
    const rawRight = Math.floor((clamp(nx2, 0, 1000) / 1000) * width);
    const rawBottom = Math.floor((clamp(ny2, 0, 1000) / 1000) * height);
 
    const boxWidth = Math.max(60, rawRight - rawLeft);
    const boxHeight = Math.max(60, rawBottom - rawTop);
 
    const padX = Math.floor(boxWidth * 0.12);
    const padY = Math.floor(boxHeight * 0.12);
 
    const left = clamp(rawLeft - padX, 0, width - 1);
    const top = clamp(rawTop - padY, 0, height - 1);
    const right = clamp(rawRight + padX, left + 1, width);
    const bottom = clamp(rawBottom + padY, top + 1, height);
 
    const extractWidth = Math.max(1, right - left);
    const extractHeight = Math.max(1, bottom - top);
 
    const cropped = await sharp(imageBuffer)
      .extract({
        left,
        top,
        width: extractWidth,
        height: extractHeight,
      })
      .resize(768, 768, {
        fit: "cover",
        position: "centre",
      })
      .png()
      .toBuffer();
 
    const croppedDataUrl = `data:image/png;base64,${cropped.toString("base64")}`;
 
    results.push({
      bucket: "innovative",
      product_handle: `innovative-${idx}-${item.title.toLowerCase().replace(/\s+/g, "-")}`,
      title: item.title,
      category: item.category,
      subcategory: item.category,
      normalized_category: item.category,
      image_url: croppedDataUrl,
      min_price: null,
      max_price: null,
      similarity: 1,
      source: "innovative",
      pinned: true,
    });
  }
 
  return results;
}
 
async function detectPinnedProducts(
  generatedImageBase64: string,
  generatedMimeType: string,
  shortlist: ShortlistItem[],
  // The pre-fetched catalogue images — used for visual matching, not just text handles.
  // Only items whose images successfully fetched are passed here, which are exactly
  // the same products that were used to furnish the generated room.
  catalogueImageParts: { inlineData: { mimeType: string; data: string } }[]
) {
  try {
    // Cap to avoid exceeding context limits — use same count as what was used for generation
    const matchShortlist = shortlist.slice(0, catalogueImageParts.length);
 
    // Build parts: generated room first, then each catalogue product image with its label
    const detectionParts: any[] = [
      { text: buildRecommendationRankingPrompt(matchShortlist) },
      { text: "Image 1 — GENERATED ROOM (identify which catalogue products appear in this room):" },
      { inlineData: { mimeType: generatedMimeType, data: generatedImageBase64 } },
    ];
 
    // Add each catalogue product image with its handle label for visual comparison
    matchShortlist.forEach((item, idx) => {
      detectionParts.push({
        text: `Catalogue product ${idx + 1} — handle="${item.product_handle}" title="${item.title}" category="${item.category ?? ""}":`,
      });
      detectionParts.push(catalogueImageParts[idx]);
    });
 
    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [{ role: "user", parts: detectionParts }],
      })
    );
 
    const text = extractTextFromResponse(response);
    const parsed = safeParseJson<RecommendationRanking>(text);
 
    const shortlistMap = new Map(shortlist.map((x) => [x.product_handle, x]));
 
    const catalogPinned: ShortlistItem[] = [];
    for (const handle of parsed.catalog_product_handles || []) {
      const item = shortlistMap.get(handle);
      if (item) {
        catalogPinned.push({ ...item, pinned: true, source: "catalog" });
      }
    }
 
    const innovativePinned = await cropInnovativeSnapshots(
      generatedImageBase64,
      generatedMimeType,
      parsed.innovative_products || []
    );
 
    return [...catalogPinned, ...innovativePinned];
  } catch (err) {
    console.warn("Failed to detect pinned products:", err);
    // Return empty on failure — better than showing wrong products as pinned
    return [];
  }
}
 
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as GenerateRoomBody;
 
    const roomType = body.roomType;
    const theme = body.theme;
    const cleanedRoomBase64Input = body.cleanedRoomBase64;
    const originalRoomBase64Input = body.originalRoomBase64 || body.cleanedRoomBase64;
    const mimeType = body.mimeType || "image/png";
    const shortlist = (body.shortlist || []).filter((x) => !!x.image_url);
    const editMode = !!body.editMode;
    const baseGeneratedImage = body.baseGeneratedImage || null;
 
    if (!roomType) {
      return NextResponse.json({ ok: false, error: "roomType is required" }, { status: 400 });
    }
 
    if (!theme?.trim()) {
      return NextResponse.json({ ok: false, error: "theme is required" }, { status: 400 });
    }
 
    if (!cleanedRoomBase64Input) {
      return NextResponse.json(
        { ok: false, error: "cleanedRoomBase64 is required" },
        { status: 400 }
      );
    }
 
    if (!shortlist.length) {
      return NextResponse.json(
        { ok: false, error: "shortlist is required and must contain image_url values" },
        { status: 400 }
      );
    }
 
    const cleanedRoomBase64 = stripDataUrlPrefix(cleanedRoomBase64Input);
    const originalRoomBase64 = stripDataUrlPrefix(originalRoomBase64Input!);
 
    let best: {
      imageBase64: string;
      mimeType: string;
      validation: FurnishedRoomValidation;
      score: number;
      shortlist: ShortlistItem[];
      usedReferenceCount: number;
      attemptNumber: number;
      catalogueImageParts: { inlineData: { mimeType: string; data: string } }[];
    } | null = null;
 
    for (let attempt = 1; attempt <= 3; attempt++) {
      const generated = await generateRoomAttempt(
        roomType,
        theme,
        cleanedRoomBase64,
        mimeType,
        shortlist,
        attempt,
        editMode,
        baseGeneratedImage
      );
 
      const validation = await validateGeneratedRoom(
        originalRoomBase64,
        mimeType,
        cleanedRoomBase64,
        mimeType,
        generated.imageBase64,
        generated.mimeType,
        roomType,
        theme,
        generated.validShortlist
      );
 
      const score = computeValidationScore(validation);
 
      if (!best || score > best.score) {
        best = {
          imageBase64: generated.imageBase64,
          mimeType: generated.mimeType,
          validation,
          score,
          shortlist: generated.validShortlist,
          usedReferenceCount: generated.usedReferenceCount,
          attemptNumber: attempt,
          catalogueImageParts: generated.catalogueImageParts,
        };
      }
    }
 
    if (!best) {
      throw new Error("Failed to generate any valid room attempt.");
    }
 
    // Pass catalogue images into detectPinnedProducts so it can visually compare
    // each product image against the generated room — far more accurate than text-only matching
    const pinnedProducts = await detectPinnedProducts(
      best.imageBase64,
      best.mimeType,
      best.shortlist,
      best.catalogueImageParts
    );
 
    return NextResponse.json({
      ok: true,
      generatedImage: `data:${best.mimeType};base64,${best.imageBase64}`,
      mimeType: best.mimeType,
      validation: best.validation,
      validationScore: best.score,
      attempts: 3,
      retryUsed: best.attemptNumber > 1,
      bestAttempt: best.attemptNumber,
      validationPassed: true,
      usedReferenceCount: best.usedReferenceCount,
      pinnedProducts,
    });
  } catch (error) {
    console.error("generate-room error:", error);
 
    return NextResponse.json(
      {
        ok: false,
        error: "Failed to generate room image",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}