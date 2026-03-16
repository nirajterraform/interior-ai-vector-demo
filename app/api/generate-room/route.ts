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

type AuthoritativeSelectionItem = ShortlistItem & {
  slot: string;
  requestedCategory: string;
  confidence?: number;
};

type ParsedIntent = {
  roomType: string;
  requestedCategories: string[];
  styleKeywords: string[];
  avoidCategories: string[];
  allowKids: boolean;
  normalizedTheme: string;
};

type EditPlanMode =
  | "initial_generate"
  | "add_object"
  | "replace_object"
  | "remove_object"
  | "restyle_object"
  | "whole_room_refresh";

type EditPlan = {
  mode: EditPlanMode;
  userRequest: string;
  targetCategory: string | null;
  preserveEverythingElse: boolean;
  allowInnovation: boolean;
  lockedProductHandles: string[];
  targetProductHandle?: string | null;
};

type SceneState = {
  catalogPinnedProducts?: ShortlistItem[];
  innovationProducts?: ShortlistItem[];
  lastEditPlan?: EditPlan | null;
};

type GenerateRoomBody = {
  roomType?: string;
  theme?: string;
  cleanedRoomBase64?: string;
  originalRoomBase64?: string;
  mimeType?: string;
  shortlist?: ShortlistItem[];
  authoritativeSelection?: AuthoritativeSelectionItem[];
  previousCatalogPinnedProducts?: ShortlistItem[];
  previousInnovationProducts?: ShortlistItem[];
  parsedIntent?: ParsedIntent | null;
  editMode?: boolean;
  baseGeneratedImage?: string | null;
  editPlan?: EditPlan | null;
  sceneState?: SceneState | null;
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

type TargetObjectDetection = {
  visible: boolean;
  title?: string;
  category?: string;
  bbox?: [number, number, number, number];
};

type GeminiImagePart = {
  inlineData: {
    mimeType: string;
    data: string;
  };
};

const SHOPIFY_IMAGE_CACHE_TTL_MS = 1000 * 60 * 15;
const SHOPIFY_IMAGE_CACHE_MAX = 250;
const MAX_GENERATION_REFERENCE_IMAGES = 14; // increased from 10 — more catalogue visuals = fewer hallucinations
const MAX_PINNED_DETECTION_REFERENCE_IMAGES = 16; // increased from 12 — covers full shortlist for detection

const shopifyImageCache = new Map<
  string,
  {
    part: GeminiImagePart;
    expiresAt: number;
    lastAccessedAt: number;
  }
>();

function evictExpiredShopifyImageEntries(now: number) {
  for (const [key, value] of shopifyImageCache.entries()) {
    if (value.expiresAt <= now) {
      shopifyImageCache.delete(key);
    }
  }
}

function evictLeastRecentlyUsedShopifyImageEntry() {
  let oldestKey: string | null = null;
  let oldestAccess = Number.POSITIVE_INFINITY;

  for (const [key, value] of shopifyImageCache.entries()) {
    if (value.lastAccessedAt < oldestAccess) {
      oldestAccess = value.lastAccessedAt;
      oldestKey = key;
    }
  }

  if (oldestKey) {
    shopifyImageCache.delete(oldestKey);
  }
}

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
  const now = Date.now();
  evictExpiredShopifyImageEntries(now);

  const cached = shopifyImageCache.get(normalizedUrl);
  if (cached && cached.expiresAt > now) {
    cached.lastAccessedAt = now;
    return cached.part;
  }

  const res = await fetch(normalizedUrl, {
    method: "GET",
    redirect: "follow",
    cache: "force-cache",
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
    .resize({ width: 640, height: 640, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 78, mozjpeg: true })
    .toBuffer();

  const part: GeminiImagePart = {
    inlineData: {
      mimeType: "image/jpeg",
      data: optimized.toString("base64"),
    },
  };

  shopifyImageCache.set(normalizedUrl, {
    part,
    expiresAt: now + SHOPIFY_IMAGE_CACHE_TTL_MS,
    lastAccessedAt: now,
  });

  if (shopifyImageCache.size > SHOPIFY_IMAGE_CACHE_MAX) {
    evictLeastRecentlyUsedShopifyImageEntry();
  }

  return part;
}

function dedupeShortlist(items: ShortlistItem[]): ShortlistItem[] {
  const seen = new Set<string>();
  const result: ShortlistItem[] = [];
  for (const item of items) {
    const handle = item?.product_handle;
    if (!handle || seen.has(handle)) continue;
    seen.add(handle);
    result.push(item);
  }
  return result;
}

function authoritativeSelectionSummary(selection: AuthoritativeSelectionItem[]) {
  return selection
    .map((item, idx) => {
      const parts = [
        `${idx + 1}.`,
        `slot=${item.slot}`,
        `requested_category=${item.requestedCategory}`,
        `product_handle=${item.product_handle}`,
        `title=${item.title}`,
        item.category ? `category=${item.category}` : "",
        item.normalized_category ? `normalized_category=${item.normalized_category}` : "",
        typeof item.confidence === "number" ? `confidence=${item.confidence.toFixed(2)}` : "",
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

    if (count >= 3 && result.length < maxItems - 2) continue; // increased from 2 — ensures 3rd-best sofa/chair still gets shown

    result.push(item);
    seen.add(handle);
    categoryCounts.set(key, count + 1);

    if (result.length >= maxItems) break;
  }

  return result;
}

function ensureSelectionCoverage(
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  theme: string
): AuthoritativeSelectionItem[] {
  if (authoritativeSelection?.length) return authoritativeSelection;
  const fallback = prioritizeShortlist(shortlist, theme, 6);
  return fallback.map((item, idx) => ({
    ...item,
    slot: idx === 0 ? "primary" : `fallback_${idx + 1}`,
    requestedCategory: item.normalized_category || item.category || item.bucket || "product",
    confidence: 0.6,
  }));
}

function normalizePromptText(text: string) {
  return String(text || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function detectTargetCategory(text: string): string | null {
  const normalized = normalizePromptText(text);
  const aliases: Array<[string, string[]]> = [
    ["sofa", ["sofa", "couch", "sectional", "loveseat", "settee"]],
    ["chair", ["chair", "armchair", "accent chair", "lounge chair", "recliner"]],
    ["table", ["table", "coffee table", "side table", "end table", "console", "nightstand"]],
    ["rug", ["rug", "carpet", "runner"]],
    ["lamp", ["lamp", "light", "floor lamp", "table lamp", "pendant", "sconce", "lighting"]],
    ["bed", ["bed", "headboard", "daybed"]],
    ["desk", ["desk", "workstation", "study table"]],
    ["cabinet", ["cabinet", "sideboard", "dresser", "wardrobe", "media unit", "bookshelf", "shelf"]],
    ["mirror", ["mirror", "wall mirror", "floor mirror"]],
    ["piano", ["piano"]],
    ["bench", ["bench"]],
    ["ottoman", ["ottoman", "pouf"]],
  ];

  for (const [canonical, words] of aliases) {
    if (words.some((word) => normalized.includes(word))) return canonical;
  }

  return null;
}

function buildServerEditPlan(
  theme: string,
  editMode: boolean,
  inputPlan: EditPlan | null | undefined,
  lockedCatalogueProducts: ShortlistItem[]
): EditPlan {
  if (inputPlan && typeof inputPlan.mode === "string") {
    return {
      ...inputPlan,
      userRequest: inputPlan.userRequest || theme,
      targetCategory: inputPlan.targetCategory || detectTargetCategory(theme),
      preserveEverythingElse: inputPlan.preserveEverythingElse ?? editMode,
      allowInnovation: inputPlan.allowInnovation ?? true,
      lockedProductHandles:
        inputPlan.lockedProductHandles?.length
          ? inputPlan.lockedProductHandles
          : lockedCatalogueProducts.map((x) => x.product_handle).filter(Boolean),
    };
  }

  const normalized = normalizePromptText(theme);
  let mode: EditPlanMode = editMode ? "whole_room_refresh" : "initial_generate";

  if (editMode) {
    if (/\b(remove|delete|without|take away)\b/.test(normalized)) mode = "remove_object";
    else if (/\b(replace|swap|change|switch)\b/.test(normalized)) mode = "replace_object";
    else if (/\b(add|include|insert|put)\b/.test(normalized)) mode = "add_object";
    else if (/\b(make|restyle|update|turn)\b/.test(normalized) && detectTargetCategory(theme)) mode = "restyle_object";
  }

  return {
    mode,
    userRequest: theme,
    targetCategory: detectTargetCategory(theme),
    preserveEverythingElse: editMode,
    allowInnovation: true,
    lockedProductHandles: lockedCatalogueProducts.map((x) => x.product_handle).filter(Boolean),
    targetProductHandle: null,
  };
}

function mergeAuthoritativeSelection(
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  lockedCatalogueProducts: ShortlistItem[],
  theme: string
) {
  const existing = new Set(authoritativeSelection.map((x) => x.product_handle));
  const merged = [...authoritativeSelection];

  for (const item of lockedCatalogueProducts) {
    if (!item?.image_url || !item.product_handle || existing.has(item.product_handle)) continue;
    merged.push({
      ...item,
      slot: `locked_${merged.length + 1}`,
      requestedCategory: item.normalized_category || item.category || item.bucket || "product",
      confidence: 0.95,
    });
    existing.add(item.product_handle);
  }

  return ensureSelectionCoverage(shortlist, merged, theme);
}


function canonicalizeCategory(value: string | null | undefined): string {
  const normalized = normalizePromptText(String(value || ""));
  if (!normalized) return "";

  const aliasMap: Array<[string, string[]]> = [
    ["sofa", ["sofa", "couch", "sectional", "loveseat", "settee"]],
    ["chair", ["chair", "armchair", "accent", "lounge chair", "recliner"]],
    ["table", ["table", "coffee table", "side table", "end table", "console", "nightstand"]],
    ["rug", ["rug", "carpet", "runner"]],
    ["lamp", ["lamp", "light", "pendant", "sconce", "lighting"]],
    ["bed", ["bed", "headboard", "daybed"]],
    ["desk", ["desk", "workstation", "study table"]],
    ["cabinet", ["cabinet", "sideboard", "dresser", "wardrobe", "media unit", "bookshelf", "shelf"]],
    ["mirror", ["mirror"]],
    ["piano", ["piano"]],
    ["bench", ["bench"]],
    ["ottoman", ["ottoman", "pouf"]],
  ];

  for (const [canonical, aliases] of aliasMap) {
    if (aliases.some((alias) => normalized.includes(alias))) return canonical;
  }

  return normalized.split(/[^a-z0-9]+/)[0] || normalized;
}

function itemCategoryKeys(item: ShortlistItem | AuthoritativeSelectionItem | null | undefined): string[] {
  if (!item) return [];
  const rawValues = [item.normalized_category, item.category, item.subcategory, item.bucket, item.title]
    .filter(Boolean)
    .map((value) => canonicalizeCategory(String(value)));
  return [...new Set(rawValues.filter(Boolean))];
}

function itemMatchesCategory(
  item: ShortlistItem | AuthoritativeSelectionItem | null | undefined,
  category: string | null | undefined
) {
  const canonical = canonicalizeCategory(category || "");
  if (!canonical) return false;
  return itemCategoryKeys(item).includes(canonical);
}

function hasCatalogueCoverage(
  category: string | null | undefined,
  items: Array<ShortlistItem | AuthoritativeSelectionItem>
) {
  const canonical = canonicalizeCategory(category || "");
  if (!canonical) return false;
  return items.some((item) => itemMatchesCategory(item, canonical));
}

function categoryCoverageSummary(items: Array<ShortlistItem | AuthoritativeSelectionItem>) {
  const categories = new Set<string>();
  for (const item of items) {
    for (const key of itemCategoryKeys(item)) categories.add(key);
  }
  return [...categories].sort().join(", ") || "none";
}

function chooseCatalogueFallbackForCategory(
  category: string | null | undefined,
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  theme: string,
  excludedHandles = new Set<string>()
): ShortlistItem | null {
  const canonical = canonicalizeCategory(category || "");
  if (!canonical) return null;

  const ranked = prioritizeShortlist(
    dedupeShortlist([
      ...authoritativeSelection,
      ...shortlist,
    ]),
    theme,
    20
  );

  for (const item of ranked) {
    if (!item?.product_handle || excludedHandles.has(item.product_handle)) continue;
    if (itemMatchesCategory(item, canonical)) return item;
  }

  return null;
}

function innovationAllowedForCategory(
  category: string | null | undefined,
  editPlan: EditPlan,
  catalogueContext: Array<ShortlistItem | AuthoritativeSelectionItem>
) {
  const canonical = canonicalizeCategory(category || "");
  if (!canonical) return editPlan.allowInnovation;

  if (hasCatalogueCoverage(canonical, catalogueContext)) {
    return false;
  }

  if (editPlan.mode === "initial_generate") return true;
  if (!editPlan.targetCategory) return true;

  return canonical === canonicalizeCategory(editPlan.targetCategory);
}

function mergePinnedProductsForRequirement(
  detectedProducts: ShortlistItem[],
  previousCatalogPinnedProducts: ShortlistItem[],
  previousInnovationProducts: ShortlistItem[],
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  theme: string,
  editPlan: EditPlan
) {
  const detectedCatalog = detectedProducts.filter((item) => item.source !== "innovative");
  const detectedInnovation = detectedProducts.filter((item) => item.source === "innovative");

  if (editPlan.mode === "initial_generate" || editPlan.mode === "whole_room_refresh") {
    return dedupeShortlist([...detectedCatalog, ...detectedInnovation]);
  }

  const targetCategory = canonicalizeCategory(editPlan.targetCategory || "");

  const keepPreviousCatalog = previousCatalogPinnedProducts.filter((item) => {
    if (!editPlan.preserveEverythingElse) return false;
    if (!targetCategory) return true;
    if (editPlan.mode === "add_object") return true;
    return !itemMatchesCategory(item, targetCategory);
  });

  const keepPreviousInnovation = previousInnovationProducts.filter((item) => {
    if (!editPlan.preserveEverythingElse) return false;
    if (!targetCategory) return true;
    if (editPlan.mode === "add_object") return true;
    return !itemMatchesCategory(item, targetCategory);
  });

  let mergedCatalog = dedupeShortlist([...keepPreviousCatalog, ...detectedCatalog]).map((item) => ({
    ...item,
    pinned: true,
    source: "catalog" as const,
  }));

  let mergedInnovation = dedupeShortlist([...keepPreviousInnovation, ...detectedInnovation]).map((item) => ({
    ...item,
    pinned: true,
    source: "innovative" as const,
  }));

  if (
    targetCategory &&
    editPlan.mode !== "remove_object" &&
    !mergedCatalog.some((item) => itemMatchesCategory(item, targetCategory)) &&
    !mergedInnovation.some((item) => itemMatchesCategory(item, targetCategory))
  ) {
    const fallback = chooseCatalogueFallbackForCategory(
      targetCategory,
      shortlist,
      authoritativeSelection,
      theme,
      new Set(mergedCatalog.map((item) => item.product_handle))
    );

    if (fallback) {
      mergedCatalog = dedupeShortlist([
        ...mergedCatalog,
        { ...fallback, pinned: true, source: "catalog" as const },
      ]);
    }
  }

  return dedupeShortlist([...mergedCatalog, ...mergedInnovation]);
}

function lockedContextSummary(items: ShortlistItem[]) {
  if (!items.length) return "None";
  return items
    .slice(0, 12)
    .map((item, idx) => `${idx + 1}. product_handle=${item.product_handle}; title=${item.title}; category=${item.normalized_category || item.category || item.bucket || "product"}`)
    .join("\n");
}

function shouldKeepLockedItemFallback(item: ShortlistItem, editPlan: EditPlan) {
  const categoryText = normalizePromptText(
    [item.normalized_category, item.category, item.bucket, item.title].filter(Boolean).join(" ")
  );
  const targetCategory = normalizePromptText(editPlan.targetCategory || "");

  if (!targetCategory) return editPlan.mode !== "whole_room_refresh";

  const targetsSameCategory = categoryText.includes(targetCategory);

  if (editPlan.mode === "remove_object" && targetsSameCategory) return false;
  if ((editPlan.mode === "replace_object" || editPlan.mode === "restyle_object") && targetsSameCategory) {
    return false;
  }

  return editPlan.preserveEverythingElse;
}

function buildSceneState(
  pinnedProducts: ShortlistItem[],
  editPlan: EditPlan
): SceneState {
  return {
    catalogPinnedProducts: pinnedProducts.filter((x) => x.source !== "innovative"),
    innovationProducts: pinnedProducts.filter((x) => x.source === "innovative"),
    lastEditPlan: editPlan,
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

function buildGeneratePrompt(
  roomType: string,
  theme: string,
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  attemptNumber: number,
  editMode: boolean,
  editPlan: EditPlan,
  lockedCatalogueProducts: ShortlistItem[],
  previousInnovationProducts: ShortlistItem[],
  coveredCatalogueCategories: string
) {
  const editText = editMode
    ? `
This is an edit request on an already generated room.
Use the current generated room as the baseline image.
Preserve every visible non-target object unless the user explicitly asked to remove or replace it.
When existing visible catalogue furniture remains in the room, preserve it instead of silently replacing or removing it.
`
    : "";

  const targetedEditText = editMode
    ? `
Edit planner contract:
- mode=${editPlan.mode}
- target_category=${editPlan.targetCategory || "none"}
- preserve_everything_else=${editPlan.preserveEverythingElse ? "true" : "false"}
- allow_innovation=${editPlan.allowInnovation ? "true" : "false"}

Locked visible catalogue products that should stay unless they are the target category:
${lockedContextSummary(lockedCatalogueProducts)}

Previously visible innovative products that should stay unless the user explicitly changes them:
${lockedContextSummary(previousInnovationProducts)}

Targeted edit rules:
1. Keep room layout, camera angle, structure, walls, floor, windows, curtains, and lighting composition stable.
2. If mode is add_object, add only the requested target object and preserve all existing visible furniture.
3. If mode is replace_object, replace only the target object/category and preserve everything else.
4. If mode is remove_object, remove only the target object/category and preserve everything else.
5. If mode is restyle_object, change only the target object/category style, color, or material and preserve everything else.
6. Do not silently replace preserved furniture with different furniture.
7. If the requested target object does not exist in catalogue, you may add it as an innovative object while preserving all locked catalogue products.
`
    : "";

  const retryText =
    attemptNumber > 1
      ? `
This is retry attempt ${attemptNumber}. Be more strict about catalog alignment, targeted editing, object identity, and preserving architecture exactly.
`
      : "";

  return `
You are an expert interior room redesign model.

Task:
Redesign this ${roomType.replaceAll("_", " ")} using the provided room image as the structural base and the provided catalogue reference products as the primary shopping source.

User design intent:
${theme}

Authoritative selected catalogue products (source of truth):
${authoritativeSelectionSummary(authoritativeSelection)}

Supporting shortlist context:
${shortlistSummary(shortlist)}

Catalogue-covered categories available right now:
${coveredCatalogueCategories}

Catalogue-authoritative generation policy:
1. Treat the authoritative selected catalogue products as the source of truth for all shoppable catalogue products used in the image.
2. For any category already covered by the shown catalogue references, you must use a catalogue item and must not invent a non-catalogue replacement.
3. Innovation is allowed only for categories that are not covered by the shown catalogue references.
4. Do not substitute, invent, replace, or introduce generic products when an authoritative selected product exists for that category.
5. Resolve vocabulary semantically. For example, couch/sectional/loveseat/settee are seating requests and should map to the closest selected seating product.
6. Never use kids or nursery furniture unless the user explicitly asks for kids, children, nursery, or toddler furniture.
7. If an authoritative selected product cannot be placed naturally, omit it rather than replacing it with another item.
8. If the user explicitly requests a new object that does not exist in catalogue, you may add it as an innovative object while preserving all existing visible catalogue furniture.
9. Never create an innovative sofa, chair, table, rug, lamp, bed, desk, cabinet, or mirror when that category is already covered by catalogue references.
10. The catalogue product images shown above are the ONLY acceptable source for those furniture items. If a sofa reference image is provided, the sofa in the generated room must visually match that reference image — not a generic sofa of a similar style. If you cannot faithfully represent the shown reference, omit that category entirely rather than substituting a different product.
11. Do not use any furniture colour, material, or silhouette that differs from the shown catalogue reference images for covered categories. The reference images define the exact visual appearance required.

Strict room-preservation rules:
10. Preserve the exact room geometry.
11. Preserve walls, windows, curtains, doors, ceiling, floor, skirting, trims, camera angle, perspective, and room proportions.
12. Add or replace only movable furniture and decor elements.
13. Do not alter architecture or add extra structural features.
14. Keep the output photorealistic and consistent with the user's requested style.
15. Keep the room type correct.
16. Use shortlist objects with strong visual alignment to the references.
17. If the user asks for color/material/style changes, apply those changes to shortlist-aligned products rather than switching to unrelated products.
${editText}
${targetedEditText}
${retryText}
Return one final redesigned room image only.
`.trim();
}

function buildValidationPrompt(roomType: string, theme: string, shortlist: ShortlistItem[], authoritativeSelection: AuthoritativeSelectionItem[]) {
  return `
You are a strict furnished-room validator.

Compare the original room, cleaned room, final furnished room, selected room type, user design prompt, and shortlisted catalog products.

Selected room type:
${roomType}

User design prompt:
${theme}

Authoritative selected catalog products:
${authoritativeSelectionSummary(authoritativeSelection)}

Supporting shortlist:
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

function buildTargetObjectDetectionPrompt(
  theme: string,
  editPlan: EditPlan,
  coveredCatalogueCategories: string
) {
  return `
You are a visual object detector for furniture editing.

You will be shown one generated room image.

User request:
${theme}

Edit planner context:
- mode=${editPlan.mode}
- target_category=${editPlan.targetCategory || "none"}
- preserve_everything_else=${editPlan.preserveEverythingElse ? "true" : "false"}

Catalogue-covered categories:
${coveredCatalogueCategories}

Task:
Determine whether the target object category requested by the user is visibly present in the generated room image.

Rules:
1. Focus on the target category only.
2. If the target category is visible, return visible=true.
3. If visible, provide a tight bbox [x1,y1,x2,y2] in 0-1000 coordinates.
4. Use the user's requested category as the category when appropriate.
5. Be conservative but do not miss an obviously visible target object.

Return strict JSON only:
{
  "visible": true,
  "title": "Piano",
  "category": "piano",
  "bbox": [120, 420, 420, 900]
}
`.trim();
}

async function detectTargetObjectIfMissing(
  theme: string,
  generatedImageBase64: string,
  generatedMimeType: string,
  editPlan: EditPlan,
  catalogueContext: ShortlistItem[],
  existingPinnedProducts: ShortlistItem[]
): Promise<ShortlistItem[]> {
  const targetCategory = canonicalizeCategory(editPlan.targetCategory || "");
  if (!targetCategory) return [];
  if (editPlan.mode === "initial_generate" || editPlan.mode === "whole_room_refresh") return [];
  if (existingPinnedProducts.some((item) => itemMatchesCategory(item, targetCategory))) return [];

  const coveredCatalogueCategories = categoryCoverageSummary(catalogueContext);

  if (hasCatalogueCoverage(targetCategory, catalogueContext)) {
    const fallback = chooseCatalogueFallbackForCategory(
      targetCategory,
      catalogueContext,
      [],
      theme,
      new Set(existingPinnedProducts.map((item) => item.product_handle))
    );
    return fallback ? [{ ...fallback, pinned: true, source: "catalog" as const }] : [];
  }

  try {
    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                text: buildTargetObjectDetectionPrompt(theme, editPlan, coveredCatalogueCategories),
              },
              { text: "Image 1: generated furnished room." },
              { inlineData: { mimeType: generatedMimeType, data: generatedImageBase64 } },
            ],
          },
        ],
      })
    );

    const parsed = safeParseJson<TargetObjectDetection>(extractTextFromResponse(response));

    if (!parsed?.visible) return [];

    const category = canonicalizeCategory(parsed.category || parsed.title || targetCategory) || targetCategory;
    const title = parsed.title || editPlan.targetCategory || "Innovative Item";
    const bbox =
      Array.isArray(parsed?.bbox) && parsed!.bbox!.length === 4
        ? (parsed!.bbox as [number, number, number, number])
        : ([200, 200, 800, 800] as [number, number, number, number]);

    const cropped = await cropInnovativeSnapshots(generatedImageBase64, generatedMimeType, [
      { title, category, bbox },
    ]);

    return cropped.map((item) => ({ ...item, pinned: true, source: "innovative" as const }));
  } catch (err) {
    console.warn("Failed to detect target object fallback:", err);
    return [];
  }
}


function buildRecommendationRankingPrompt(
  theme: string,
  shortlist: ShortlistItem[],
  editPlan: EditPlan | undefined,
  coveredCatalogueCategories: string
) {
  return `
You are a visual product matching expert.

You will be shown:
- Image 1: the generated furnished room
- Images 2 onwards: catalogue product reference images, each labelled with its product_handle

User edit / design request:
${theme}

Edit planner context:
- mode=${editPlan?.mode || "unknown"}
- target_category=${editPlan?.targetCategory || "none"}
- preserve_everything_else=${editPlan?.preserveEverythingElse ? "true" : "false"}

Catalogue-covered categories in the shown references:
${coveredCatalogueCategories}

Tasks:
1. Return catalog_product_handles only for catalogue products clearly visible in the generated room.
2. Check every visible furniture/decor object against the full set of shown catalogue references before deciding it is not from the catalogue.
3. If a visible object belongs to a category that is covered by the shown catalogue references, prefer mapping it to the closest catalogue product_handle instead of declaring it innovative.
4. Return innovative_products only for large visible objects that do not match any shown catalogue product and belong to categories not covered by the shown catalogue references.
5. If the user explicitly asked for a new or changed object such as piano, bench, bookshelf, mirror, vanity, pouf, ottoman, or sofa and it is visible but not in the shown references, include it in innovative_products only when that category is truly not covered by the shown catalogue references.
6. Do not omit a shown catalogue product if it is visibly present in the generated room.
7. For each innovative object, return a tight bbox using [x1,y1,x2,y2] in 0-1000 coordinates.
8. When preserve_everything_else is true, try hard to return every still-visible catalogue product, even if the current edit only changed one object.
9. Never classify a visible sofa/chair/table/rug/lamp/bed/desk/cabinet/mirror as innovative when a shown catalogue reference exists for that category.

Catalogue product handles:
${shortlistSummary(shortlist)}

Return strict JSON only:
{
  "catalog_product_handles": ["handle1"],
  "innovative_products": [
    {
      "title": "Piano",
      "category": "piano",
      "bbox": [120, 420, 420, 900]
    }
  ]
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

function isStrongEnoughToStop(v: FurnishedRoomValidation, score: number) {
  return (
    v.pass &&
    v.room_type_correct &&
    v.geometry_preserved &&
    v.catalog_alignment_level !== "poor" &&
    v.theme_match_level !== "poor" &&
    v.hallucinated_objects_level !== "significant" &&
    v.artifacts_level !== "significant" &&
    score >= 28
  );
}

async function buildReferenceParts(shortlist: ShortlistItem[], authoritativeSelection: AuthoritativeSelectionItem[], theme: string) {
  const prioritizedSelection = authoritativeSelection?.length
    ? authoritativeSelection
    : ensureSelectionCoverage(shortlist, authoritativeSelection, theme);
  const prioritized = prioritizeShortlist(prioritizedSelection, theme, MAX_GENERATION_REFERENCE_IMAGES);

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
  authoritativeSelection: AuthoritativeSelectionItem[],
  attemptNumber: number,
  editMode: boolean,
  editPlan: EditPlan,
  lockedCatalogueProducts: ShortlistItem[],
  previousInnovationProducts: ShortlistItem[],
  baseGeneratedImage?: string | null
) {
  const effectiveSelection = ensureSelectionCoverage(shortlist, authoritativeSelection, theme);
  const { referenceParts, validShortlist, catalogueImageParts } = await buildReferenceParts(shortlist, effectiveSelection, theme);
  if (!referenceParts.length) {
    throw new Error("None of the catalogue images could be fetched from Shopify CDN.");
  }

  const selectedForPrompt = ensureSelectionCoverage(validShortlist, effectiveSelection, theme);
  const coveredCatalogueCategories = categoryCoverageSummary([
    ...selectedForPrompt,
    ...validShortlist,
    ...lockedCatalogueProducts,
  ]);

  const prompt = buildGeneratePrompt(
    roomType,
    theme,
    validShortlist,
    selectedForPrompt,
    attemptNumber,
    editMode,
    editPlan,
    lockedCatalogueProducts,
    previousInnovationProducts,
    coveredCatalogueCategories
  );
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
    parts.push({ text: "Cleaned room base image:" });
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
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      },
    })
  );

  const image = extractImageFromResponse(response);
  const rawText = extractTextFromResponse(response) || null;

  if (!image) {
    throw new Error(
      `Room generation did not return an image.${rawText ? ` Model text: ${rawText}` : ""}`
    );
  }

  return {
    imageBase64: image.data,
    mimeType: image.mimeType,
    rawText,
    validShortlist,
    catalogueImageParts,
    usedReferenceCount: referenceParts.length,
    authoritativeSelection: selectedForPrompt,
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
  shortlist: ShortlistItem[],
  authoritativeSelection: AuthoritativeSelectionItem[]
): Promise<FurnishedRoomValidation> {
  const response = await withGeminiRetry(() =>
    ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        {
          role: "user",
          parts: [
            { text: buildValidationPrompt(roomType, theme, shortlist, authoritativeSelection) },
            { text: "Image 1: original room image." },
            { inlineData: { mimeType: originalMimeType, data: originalRoomBase64 } },
            { text: "Image 2: cleaned room image." },
            { inlineData: { mimeType: cleanedMimeType, data: cleanedRoomBase64 } },
            { text: "Image 3: final furnished room image." },
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

  const safeSlug =
    safeTitle
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

function buildPinnedDetectionReferences(
  shortlist: ShortlistItem[],
  catalogueImageParts: GeminiImagePart[],
  authoritativeSelection: AuthoritativeSelectionItem[]
) {
  const authoritativeHandles = new Set(
    (authoritativeSelection || []).map((item) => item.product_handle).filter(Boolean)
  );

  const pairs = shortlist
    .map((item, index) => ({
      item,
      part: catalogueImageParts[index],
      priority: authoritativeHandles.has(item.product_handle) ? 1 : 0,
      originalIndex: index,
    }))
    .filter((pair) => !!pair.item?.product_handle && !!pair.part);

  pairs.sort((a, b) => {
    if (b.priority !== a.priority) return b.priority - a.priority;
    return a.originalIndex - b.originalIndex;
  });

  const limited = pairs.slice(0, MAX_PINNED_DETECTION_REFERENCE_IMAGES);

  return {
    referenceShortlist: limited.map((pair) => pair.item),
    limitedCatalogueImageParts: limited.map((pair) => pair.part),
  };
}

async function detectPinnedProducts(
  theme: string,
  generatedImageBase64: string,
  generatedMimeType: string,
  shortlist: ShortlistItem[],
  catalogueImageParts: GeminiImagePart[],
  authoritativeSelection: AuthoritativeSelectionItem[],
  editPlan: EditPlan,
  previousCatalogPinnedProducts: ShortlistItem[],
  previousInnovationProducts: ShortlistItem[]
): Promise<ShortlistItem[]> {
  try {
    const { referenceShortlist, limitedCatalogueImageParts } = buildPinnedDetectionReferences(
      shortlist,
      catalogueImageParts,
      authoritativeSelection
    );

    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                text: buildRecommendationRankingPrompt(
                  theme,
                  referenceShortlist,
                  editPlan,
                  categoryCoverageSummary([
                    ...referenceShortlist,
                    ...authoritativeSelection,
                    ...previousCatalogPinnedProducts,
                  ])
                ),
              },
              { text: "Image 1: generated furnished room." },
              { inlineData: { mimeType: generatedMimeType, data: generatedImageBase64 } },
              ...limitedCatalogueImageParts.flatMap((part, index) => [
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
    const catalogueContext = dedupeShortlist([
      ...authoritativeSelection,
      ...shortlist,
      ...previousCatalogPinnedProducts,
    ]);

    let catalogPinned: ShortlistItem[] = [];
    for (const handle of parsed.catalog_product_handles || []) {
      const item = shortlistMap.get(handle) || catalogueContext.find((candidate) => candidate.product_handle === handle);
      if (item) catalogPinned.push({ ...item, pinned: true, source: "catalog" });
    }

    let innovativePinned = await cropInnovativeSnapshots(
      generatedImageBase64,
      generatedMimeType,
      parsed.innovative_products || []
    );

    const detectedCatalogHandles = new Set(catalogPinned.map((item) => item.product_handle));
    const filteredInnovation: ShortlistItem[] = [];

    for (const innovativeItem of innovativePinned) {
      const innovationCategory = canonicalizeCategory(
        innovativeItem.normalized_category || innovativeItem.category || innovativeItem.title
      );

      if (!innovationAllowedForCategory(innovationCategory, editPlan, catalogueContext)) {
        const fallback = chooseCatalogueFallbackForCategory(
          innovationCategory,
          shortlist,
          authoritativeSelection,
          theme,
          detectedCatalogHandles
        );

        if (fallback && !detectedCatalogHandles.has(fallback.product_handle)) {
          catalogPinned.push({ ...fallback, pinned: true, source: "catalog" });
          detectedCatalogHandles.add(fallback.product_handle);
        }
        continue;
      }

      filteredInnovation.push({ ...innovativeItem, pinned: true, source: "innovative" });
    }

    let merged = mergePinnedProductsForRequirement(
      dedupeShortlist([...catalogPinned, ...filteredInnovation]),
      previousCatalogPinnedProducts,
      previousInnovationProducts,
      shortlist,
      authoritativeSelection,
      theme,
      editPlan
    );

    const targetFallbackItems = await detectTargetObjectIfMissing(
      theme,
      generatedImageBase64,
      generatedMimeType,
      editPlan,
      catalogueContext,
      merged
    );

    if (targetFallbackItems.length) {
      merged = dedupeShortlist([...merged, ...targetFallbackItems]);
    }

    return dedupeShortlist(merged);
  } catch (err) {
    console.warn("Failed to detect pinned products:", err);
    return mergePinnedProductsForRequirement(
      [],
      previousCatalogPinnedProducts,
      previousInnovationProducts,
      shortlist,
      authoritativeSelection,
      theme,
      editPlan
    );
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
    const previousCatalogPinnedProducts = (body.previousCatalogPinnedProducts || body.sceneState?.catalogPinnedProducts || []).filter((x) => !!x?.image_url);
    const previousInnovationProducts = (body.previousInnovationProducts || body.sceneState?.innovationProducts || []).filter((x) => !!x);
    const mergedShortlist = dedupeShortlist([...shortlist, ...previousCatalogPinnedProducts]);
    const editMode = !!body.editMode;
    const editPlan = buildServerEditPlan(theme || "", editMode, body.editPlan, previousCatalogPinnedProducts);
    const authoritativeSelection = mergeAuthoritativeSelection(
      mergedShortlist,
      (body.authoritativeSelection || []).filter((x) => !!x?.image_url),
      previousCatalogPinnedProducts,
      theme || ""
    );
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
    if (!mergedShortlist.length) {
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
      authoritativeSelection: AuthoritativeSelectionItem[];
    } | null = null;

    for (let attempt = 1; attempt <= 2; attempt++) {
      const generated = await generateRoomAttempt(
        roomType,
        theme,
        cleanedRoomBase64,
        mimeType,
        mergedShortlist,
        authoritativeSelection,
        attempt,
        editMode,
        editPlan,
        previousCatalogPinnedProducts,
        previousInnovationProducts,
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
        generated.validShortlist,
        generated.authoritativeSelection
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
          authoritativeSelection: generated.authoritativeSelection,
        };
      }

      if (isStrongEnoughToStop(validation, score)) {
        break;
      }

      if (
        attempt === 1 &&
        validation.pass &&
        validation.catalog_alignment_level !== "poor" &&
        validation.theme_match_level !== "poor" &&
        validation.structural_drift_level !== "significant"
      ) {
        break;
      }
    }

    if (!best) {
      return NextResponse.json({ ok: false, error: "Room generation failed" }, { status: 500 });
    }

    const pinnedProducts = await detectPinnedProducts(
      theme,
      best.imageBase64,
      best.mimeType,
      best.shortlist,
      best.catalogueImageParts,
      best.authoritativeSelection,
      editPlan,
      previousCatalogPinnedProducts,
      previousInnovationProducts
    );
    const sceneState = buildSceneState(pinnedProducts, editPlan);

    return NextResponse.json({
      ok: true,
      generatedImage: `data:${best.mimeType};base64,${best.imageBase64}`,
      pinnedProducts,
      selectedProducts: best.authoritativeSelection,
      validationPassed: best.validation.pass,
      validation: best.validation,
      usedReferenceCount: best.usedReferenceCount,
      attemptNumber: best.attemptNumber,
      editPlan,
      sceneState,
    });
  } catch (error) {
    console.error("generate-room error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to generate room" },
      { status: 500 }
    );
  }
}
