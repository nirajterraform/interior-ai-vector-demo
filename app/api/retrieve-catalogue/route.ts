import { NextRequest, NextResponse } from "next/server";
import { retrieveCatalogue } from "@/lib/retrieval";

type RetrievedProduct = {
  bucket: string;
  product_handle: string;
  title: string;
  category?: string | null;
  subcategory?: string | null;
  normalized_category?: string | null;
  image_url?: string | null;
  min_price?: number | null;
  max_price?: number | null;
  similarity?: number;
};

const CATEGORY_ALIASES: Record<string, string[]> = {
  sofa: ["sofa", "couch", "sectional", "settee", "loveseat", "chesterfield"],
  chair: ["chair", "armchair", "accent chair", "lounge chair", "recliner", "easy chair"],
  table: ["table", "coffee table", "side table", "end table", "center table", "console", "nightstand", "bedside table"],
  rug: ["rug", "carpet", "runner"],
  lamp: ["lamp", "light", "floor lamp", "table lamp", "pendant", "sconce", "lighting"],
  bed: ["bed", "headboard", "daybed", "bunk bed"],
  cabinet: ["cabinet", "sideboard", "storage", "credenza", "dresser", "wardrobe", "media unit", "tv unit", "bookshelf", "shelf"],
  desk: ["desk", "workstation", "study table", "office table"],
  stool: ["stool", "bar stool", "ottoman", "pouf", "bench"],
  dining: ["dining", "dining chair", "dining table"],
  mirror: ["mirror", "wall mirror", "floor mirror"],
  decor: ["decor", "art", "wall art", "vase", "planter", "plant", "accessory", "cushion", "throw pillow"],
};

const ROOM_TYPE_HINTS: Record<string, string[]> = {
  living_room: ["sofa", "chair", "table", "rug", "lamp", "cabinet", "mirror", "decor", "stool"],
  bedroom: ["bed", "table", "lamp", "cabinet", "mirror", "rug", "chair", "stool", "decor"],
  dining_room: ["dining", "chair", "table", "lamp", "cabinet", "rug", "decor"],
  kitchen: ["stool", "table", "lamp", "cabinet", "decor"],
  office: ["desk", "chair", "lamp", "cabinet", "rug", "decor"],
};

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeWhitespace(s: string) {
  return s.replace(/\s+/g, " ").trim();
}

function normalizeTheme(theme: string): string {
  let result = theme.toLowerCase();

  for (const [canonical, aliases] of Object.entries(CATEGORY_ALIASES)) {
    for (const alias of aliases) {
      const re = new RegExp(`\\b${escapeRegExp(alias)}\\b`, "gi");
      result = result.replace(re, canonical);
    }
  }

  return normalizeWhitespace(result);
}

function extractRequestedCategories(theme: string, roomType: string): string[] {
  const text = normalizeTheme(theme);
  const requested: string[] = [];

  for (const [canonical, aliases] of Object.entries(CATEGORY_ALIASES)) {
    const words = [canonical, ...aliases];
    if (words.some((word) => new RegExp(`\\b${escapeRegExp(word.toLowerCase())}\\b`, "i").test(text))) {
      requested.push(canonical);
    }
  }

  if (!requested.length) {
    return ROOM_TYPE_HINTS[roomType] || [];
  }

  const merged = new Set<string>(requested);
  for (const fallback of ROOM_TYPE_HINTS[roomType] || []) {
    if (merged.size >= 6) break;
    merged.add(fallback);
  }
  return [...merged];
}

function shouldAllowKids(theme: string, roomType: string) {
  if (roomType === "bedroom" && /\bkid|kids|children|child|nursery|toddler\b/i.test(theme)) {
    return true;
  }
  return /\bkid|kids|children|child|nursery|toddler\b/i.test(theme);
}

function itemSearchText(item: RetrievedProduct) {
  return normalizeWhitespace(
    [
      item.title,
      item.category,
      item.subcategory,
      item.normalized_category,
      item.bucket,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase()
  );
}

function scoreProduct(item: RetrievedProduct, requestedCategories: string[], theme: string, roomType: string) {
  const text = itemSearchText(item);
  let score = Number(item.similarity || 0);

  const allowKids = shouldAllowKids(theme, roomType);
  if (!allowKids && /\bkid|kids|children|child|nursery|toddler\b/.test(text)) {
    score -= 100;
  }

  for (const category of requestedCategories) {
    if (text.includes(category)) {
      score += 1.25;
    }
  }

  const roomHints = ROOM_TYPE_HINTS[roomType] || [];
  if (roomHints.some((hint) => text.includes(hint))) {
    score += 0.2;
  }

  if (roomType !== "bedroom" && /\bbed|headboard|bunk bed\b/.test(text)) score -= 2;
  if (roomType !== "office" && /\bdesk|workstation\b/.test(text) && !requestedCategories.includes("desk")) score -= 0.5;
  if (roomType !== "dining_room" && /\bdining\b/.test(text) && !requestedCategories.includes("dining")) score -= 0.5;

  return score;
}

function filterAndPrioritize(
  items: RetrievedProduct[],
  requestedCategories: string[],
  theme: string,
  roomType: string,
  maxItems: number
) {
  const scored = items
    .map((item) => ({
      item,
      text: itemSearchText(item),
      score: scoreProduct(item, requestedCategories, theme, roomType),
    }))
    .filter((entry) => Number.isFinite(entry.score) && entry.score > -50)
    .sort((a, b) => b.score - a.score);

  const result: RetrievedProduct[] = [];
  const usedHandles = new Set<string>();
  const categoryCount = new Map<string, number>();

  const requestSet = new Set(requestedCategories);

  for (const entry of scored) {
    const item = entry.item;
    if (usedHandles.has(item.product_handle)) continue;

    const key = (item.normalized_category || item.category || item.bucket || "other").toLowerCase();
    const currentCount = categoryCount.get(key) || 0;
    const isRequestedFamily = [...requestSet].some((cat) => entry.text.includes(cat));

    if (currentCount >= 2 && !isRequestedFamily && result.length < maxItems - 2) {
      continue;
    }

    result.push(item);
    usedHandles.add(item.product_handle);
    categoryCount.set(key, currentCount + 1);

    if (result.length >= maxItems) break;
  }

  return result;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const roomType = body?.roomType;
    const theme = String(body?.theme || "").trim();
    const seenHandles = Array.isArray(body?.seenHandles) ? body.seenHandles : [];
    const rotationCursor = typeof body?.rotationCursor === "number" ? body.rotationCursor : 0;
    const pageSize = typeof body?.pageSize === "number" && body.pageSize > 0 ? Math.min(body.pageSize, 18) : 12;

    if (!roomType) {
      return NextResponse.json({ error: "roomType is required" }, { status: 400 });
    }

    if (!theme) {
      return NextResponse.json({ error: "theme is required" }, { status: 400 });
    }

    const normalizedTheme = normalizeTheme(theme);
    const requestedCategories = extractRequestedCategories(theme, roomType);

    const expandedQuery = normalizeWhitespace(
      [normalizedTheme, ...requestedCategories].join(" ")
    );

    const result = await retrieveCatalogue({
      roomType,
      theme: expandedQuery,
      seenHandles,
      rotationCursor,
      pageSize: Math.max(pageSize, 12),
    });

    const shortlist = filterAndPrioritize(
      result.shortlist || [],
      requestedCategories,
      theme,
      roomType,
      pageSize
    );

    return NextResponse.json({
      ...result,
      theme,
      normalizedTheme,
      requestedCategories,
      shortlist,
    });
  } catch (error) {
    console.error("retrieve-catalogue error:", error);

    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to retrieve catalogue",
      },
      { status: 500 }
    );
  }
}
