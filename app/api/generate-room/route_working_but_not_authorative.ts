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

type GeminiImagePart = {
  inlineData: {
    mimeType: string;
    data: string;
  };
};

function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}

function extractTextFromResponse(response: any): string {
  const parts = response?.candidates?.[0]?.content?.parts || [];
  return parts
    .filter((p: any) => typeof p?.text === "string")
    .map((p: any) => p.text)
    .join("")
    .trim();
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
    if (match) return JSON.parse(match[0]) as T;
    throw new Error(`No valid JSON object found in model output: ${cleaned.slice(0, 1200)}`);
  }
}

async function urlToInlineData(url: string): Promise<GeminiImagePart> {
  const normalizedUrl = decodeURI(url.trim());
  const res = await fetch(normalizedUrl, {
    method: "GET",
    redirect: "follow",
    cache: "no-store",
    headers: {
      "User-Agent": "Mozilla/5.0",
      Accept: "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
      Referer: "https://www.shopify.com/",
    },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch catalogue image: ${normalizedUrl} (status ${res.status})`);
  }

  const arrayBuffer = await res.arrayBuffer();
  const optimized = await sharp(Buffer.from(arrayBuffer))
    .rotate()
    .resize({ width: 768, height: 768, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 82, mozjpeg: true })
    .toBuffer();

  return {
    inlineData: {
      mimeType: "image/jpeg",
      data: optimized.toString("base64"),
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
        item.normalized_category ? `normalized_category=${item.normalized_category}` : "",
      ].filter(Boolean);
      return parts.join("; ");
    })
    .join("\n");
}

function scoreCategoryFit(item: ShortlistItem, theme: string) {
  const text = [item.title, item.category, item.subcategory, item.normalized_category, item.bucket]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  const query = theme.toLowerCase();
  let score = Number(item.similarity || 0);

  const intentMap: Array<[RegExp, string[]]> = [
    [/\bsofa|couch|sectional|loveseat|settee\b/i, ["sofa", "couch", "sectional", "loveseat", "settee"]],
    [/\bchair|armchair|accent chair|lounge chair\b/i, ["chair", "armchair", "accent"]],
    [/\btable|coffee table|side table|end table|console\b/i, ["table", "coffee table", "side table", "console"]],
    [/\brug|carpet|runner\b/i, ["rug", "carpet", "runner"]],
    [/\blamp|light|pendant\b/i, ["lamp", "light", "pendant"]],
    [/\bbed|daybed|headboard\b/i, ["bed", "daybed", "headboard"]],
  ];

  for (const [pattern, aliases] of intentMap) {
    if (pattern.test(query)) {
      if (aliases.some((a) => text.includes(a))) score += 0.75;
      else score -= 0.2;
    }
  }

  if (
    !/\bkid|kids|children|child|nursery|toddler\b/i.test(query) &&
    /\bkid|kids|children|child|nursery|toddler\b/i.test(text)
  ) {
    score -= 1.5;
  }

  return score;
}

function prioritizeShortlist(shortlist: ShortlistItem[], theme: string, maxItems = 8) {
  const scored = [...shortlist].sort(
    (a, b) => scoreCategoryFit(b, theme) - scoreCategoryFit(a, theme)
  );

  const result: ShortlistItem[] = [];
  const seen = new Set<string>();
  const categoryCounts = new Map<string, number>();

  for (const item of scored) {
    const handle = item.product_handle;
    if (!handle || seen.has(handle)) continue;

    const key = (item.normalized_category || item.category || item.bucket || "other").toLowerCase();
    const count = categoryCounts.get(key) || 0;

    if (count >= 2 && result.length < maxItems - 2) continue;

    result.push(item);
    seen.add(handle);
    categoryCounts.set(key, count + 1);

    if (result.length >= maxItems) break;
  }

  return result;
}

function buildGeneratePrompt(
  roomType: string,
  theme: string,
  shortlist: ShortlistItem[],
  attemptNumber: number,
  editMode: boolean
) {
  const editText = editMode
    ? `\nThis is an edit request on an already generated room.\nApply only the requested changes while preserving the rest of the room as much as possible.\n`
    : "";

  const retryText =
    attemptNumber > 1
      ? `\nThis is retry attempt ${attemptNumber}. Be more strict about catalog alignment, object identity, and preserving architecture exactly.\n`
      : "";

  return `
You are an expert interior room redesign model.

Task:
Redesign this ${roomType.replaceAll("_", " ")} using the cleaned room image as the structural base and the provided catalogue reference products as the only shopping source.

User design intent:
${theme}

Catalogue references:
${shortlistSummary(shortlist)}

Catalogue-aware generation policy:
1. Treat the catalogue shortlist as the source of truth for shoppable products.
2. If a matching catalogue product exists for a requested object, use that catalogue product family and do not invent an internet or generic substitute.
3. Resolve vocabulary semantically. For example, couch/sectional/loveseat/settee are seating requests and should map to the closest seating item in the shortlist.
4. Never use kids or nursery furniture unless the user explicitly asks for kids, children, nursery, or toddler furniture.
5. If no suitable shortlist product exists for a requested object, omit that object instead of inventing a mismatched one.

Strict room-preservation rules:
6. Preserve the exact room geometry.
7. Preserve walls, windows, curtains, doors, ceiling, floor, skirting, trims, camera angle, perspective, and room proportions.
8. Add or replace only movable furniture and decor elements.
9. Do not alter architecture or add extra structural features.
10. Keep the output photorealistic and consistent with the user's requested style.
11. Keep the room type correct.
12. Use shortlist objects with strong visual alignment to the references.
13. If the user asks for color/material/style changes, apply those changes to shortlist-aligned products rather than switching to unrelated products.
${editText}
${retryText}
Return one final redesigned room image only.
`.trim();
}

function buildValidationPrompt(roomType: string, theme: string, shortlist: ShortlistItem[]) {
  return `
You are a strict furnished-room validator.

Compare the original room, cleaned room, final furnished room, selected room type, user design prompt, and shortlisted catalog products.

Selected room type:
${roomType}

User design prompt:
${theme}

Shortlisted catalog products:
${shortlistSummary(shortlist)}

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

Tasks:
1. Return catalog_product_handles only for products clearly visible in the generated room.
2. Return innovative_products only for large objects visible in the room that do not match any shown catalogue product.
3. Be conservative. If uncertain, do not mark it as pinned.

Catalogue product handles:
${shortlistSummary(shortlist)}

Return strict JSON only:
{
  "catalog_product_handles": ["handle1"],
  "innovative_products": []
}
`.trim();
}

function levelScore(level: Level): number {
  return level === "none" ? 3 : level === "minor" ? 2 : level === "moderate" ? 1 : 0;
}

function qualityScore(q: Quality): number {
  return q === "excellent" ? 4 : q === "good" ? 3 : q === "acceptable" ? 2 : 0;
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

async function buildReferenceParts(shortlist: ShortlistItem[], theme: string) {
  const prioritized = prioritizeShortlist(shortlist, theme, 8);

  const resolved = await Promise.all(
    prioritized.map(async (item) => {
      try {
        const part = await urlToInlineData(item.image_url as string);
        return { ok: true as const, part, item };
      } catch (err: any) {
        const status = String(err?.message ?? "").match(/status (\d+)/)?.[1];
        if (status !== "404") {
          console.warn("Skipping catalogue image:", item.image_url, status ?? String(err));
        }
        return { ok: false as const, part: null, item };
      }
    })
  );

  const fetched = resolved.filter((x) => x.ok && x.part !== null);
  return {
    referenceParts: fetched.map((x) => x.part),
    validShortlist: fetched.map((x) => x.item),
    catalogueImageParts: fetched.map((x) => x.part) as GeminiImagePart[],
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
  const { referenceParts, validShortlist, catalogueImageParts } = await buildReferenceParts(shortlist, theme);
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
      contents: [{ role: "user", parts }],
      config: { responseModalities: [Modality.IMAGE, Modality.TEXT] },
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
            { text: buildValidationPrompt(roomType, theme, shortlist.slice(0, 8)) },
            { text: "Image 1: original room image." },
            { inlineData: { mimeType: originalMimeType, data: originalRoomBase64 } },
            { text: "Image 2: cleaned room image." },
            { inlineData: { mimeType: cleanedMimeType, data: cleanedRoomBase64 } },
            { text: "Image 3: final furnished generated room image." },
            { inlineData: { mimeType: generatedMimeType, data: generatedImageBase64 } },
          ],
        },
      ],
    })
  );

  return safeParseJson<FurnishedRoomValidation>(extractTextFromResponse(response));
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}


function normalizeInnovativeItem(
  item: Partial<InnovativeProductDetection> | null | undefined,
  idx: number
) {
  const safeTitle =
    typeof item?.title === "string" && item.title.trim().length > 0
      ? item.title.trim()
      : typeof item?.category === "string" && item.category.trim().length > 0
      ? item.category.trim()
      : `Innovative Item ${idx + 1}`;

  const safeCategory =
    typeof item?.category === "string" && item.category.trim().length > 0
      ? item.category.trim()
      : "innovative";

  const safeSlug = safeTitle
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || `innovative-item-${idx + 1}`;

  const bbox =
    Array.isArray(item?.bbox) && item!.bbox.length === 4
      ? (item!.bbox as [number, number, number, number])
      : ([200, 200, 800, 800] as [number, number, number, number]);

  return { safeTitle, safeCategory, safeSlug, bbox };
}

async function cropInnovativeSnapshots(
  generatedImageBase64: string,
  generatedMimeType: string,
  innovativeProducts: InnovativeProductDetection[]
): Promise<ShortlistItem[]> {
  const cleanedInnovativeProducts = (innovativeProducts || []).filter(
    (item) => item && (typeof item.title === "string" || typeof item.category === "string")
  );

  const imageBuffer = Buffer.from(generatedImageBase64, "base64");
  const image = sharp(imageBuffer);
  const metadata = await image.metadata();
  const width = metadata.width ?? 0;
  const height = metadata.height ?? 0;

  if (!width || !height) {
    return cleanedInnovativeProducts.map((item, idx) => {
      const normalized = normalizeInnovativeItem(item, idx);
      return {
        bucket: "innovative",
        product_handle: `innovative-${idx}-${normalized.safeSlug}`,
        title: normalized.safeTitle,
        category: normalized.safeCategory,
        subcategory: normalized.safeCategory,
        normalized_category: normalized.safeCategory,
        image_url: null,
        min_price: null,
        max_price: null,
        similarity: 1,
        source: "innovative",
        pinned: true,
      };
    });
  }

  const results: ShortlistItem[] = [];
  for (let idx = 0; idx < cleanedInnovativeProducts.length; idx++) {
    const item = cleanedInnovativeProducts[idx];
    const normalized = normalizeInnovativeItem(item, idx);
    const [nx1, ny1, nx2, ny2] = normalized.bbox;

    const left = clamp(Math.round((nx1 / 1000) * width), 0, width - 1);
    const top = clamp(Math.round((ny1 / 1000) * height), 0, height - 1);
    const right = clamp(Math.round((nx2 / 1000) * width), left + 1, width);
    const bottom = clamp(Math.round((ny2 / 1000) * height), top + 1, height);
    const cropWidth = Math.max(1, right - left);
    const cropHeight = Math.max(1, bottom - top);

    const cropBuffer = await image
      .clone()
      .extract({ left, top, width: cropWidth, height: cropHeight })
      .jpeg({ quality: 84 })
      .toBuffer();

    results.push({
      bucket: "innovative",
      product_handle: `innovative-${idx}-${normalized.safeSlug}`,
      title: normalized.safeTitle,
      category: normalized.safeCategory,
      subcategory: normalized.safeCategory,
      normalized_category: normalized.safeCategory,
      image_url: `data:image/jpeg;base64,${cropBuffer.toString("base64")}`,
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
  catalogueImageParts: GeminiImagePart[]
): Promise<ShortlistItem[]> {
  try {
    const referenceShortlist = shortlist.slice(0, catalogueImageParts.length);

    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
            role: "user",
            parts: [
              { text: buildRecommendationRankingPrompt(referenceShortlist) },
              { text: "Image 1: generated furnished room." },
              { inlineData: { mimeType: generatedMimeType, data: generatedImageBase64 } },
              ...catalogueImageParts.flatMap((part, index) => [
                {
                  text: `Reference image ${index + 2}: product_handle=${
                    referenceShortlist[index]?.product_handle || "unknown"
                  }`,
                },
                part,
              ]),
            ],
          },
        ],
      })
    );

    const parsed = safeParseJson<RecommendationRanking>(extractTextFromResponse(response));
    const shortlistMap = new Map(shortlist.map((x) => [x.product_handle, x]));

    const catalogPinned: ShortlistItem[] = [];
    for (const handle of parsed.catalog_product_handles || []) {
      const item = shortlistMap.get(handle);
      if (item) catalogPinned.push({ ...item, pinned: true, source: "catalog" });
    }

    const innovativePinned = await cropInnovativeSnapshots(
      generatedImageBase64,
      generatedMimeType,
      parsed.innovative_products || []
    );

    return [...catalogPinned, ...innovativePinned];
  } catch (err) {
    console.warn("Failed to detect pinned products:", err);
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
      return NextResponse.json({ ok: false, error: "cleanedRoomBase64 is required" }, { status: 400 });
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
      catalogueImageParts: GeminiImagePart[];
    } | null = null;

    for (let attempt = 1; attempt <= 2; attempt++) {
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

      if (
        validation.pass &&
        validation.catalog_alignment_level !== "poor" &&
        validation.theme_match_level !== "poor"
      ) {
        break;
      }
    }

    if (!best) {
      return NextResponse.json({ ok: false, error: "Room generation failed" }, { status: 500 });
    }

    const pinnedProducts = await detectPinnedProducts(
      best.imageBase64,
      best.mimeType,
      best.shortlist,
      best.catalogueImageParts
    );

    return NextResponse.json({
      ok: true,
      generatedImage: `data:${best.mimeType};base64,${best.imageBase64}`,
      pinnedProducts,
      validationPassed: best.validation.pass,
      validation: best.validation,
      usedReferenceCount: best.usedReferenceCount,
      attemptNumber: best.attemptNumber,
    });
  } catch (error) {
    console.error("generate-room error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to generate room" },
      { status: 500 }
    );
  }
}
