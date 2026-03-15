import { db } from "@/lib/db";
import { embedQuery } from "@/lib/embeddings";
import { RetrievedProduct, RoomType } from "@/lib/types";

type BucketConfig = {
  bucket:
    | "seating"
    | "tables"
    | "lighting"
    | "wall_art"
    | "decor"
    | "storage"
    | "soft_furnishing"
    | "bed";
  limit: number;
  required: boolean;
  allowedCategories?: string[];
};

type RetrieveCatalogueParams = {
  roomType: RoomType;
  theme: string;
  seenHandles?: string[];
  rotationCursor?: number;
  pageSize?: number;
};

type RetrievalResult = {
  roomType: RoomType;
  theme: string;
  shortlist: RetrievedProduct[];
  nextRotationCursor: number;
};

const ROOM_BUCKETS: Record<RoomType, BucketConfig[]> = {
  living_room: [
    {
      bucket: "seating",
      limit: 8,
      required: true,
      allowedCategories: ["sofa", "accent_chair", "ottoman", "bench"],
    },
    {
      bucket: "tables",
      limit: 4,
      required: true,
      allowedCategories: ["coffee_table", "side_table"],
    },
    {
      bucket: "lighting",
      limit: 2,
      required: true,
      allowedCategories: ["lamp"],
    },
    {
      bucket: "wall_art",
      limit: 4,
      required: false,
      allowedCategories: [
        "framed_art",
        "canvas_art",
        "canvas_sign",
        "metal_sign",
        "map_art",
        "wall_hanging",
      ],
    },
    {
      bucket: "decor",
      limit: 4,
      required: false,
      allowedCategories: ["decor", "mirror", "artificial_plant", "floral_arrangement"],
    },
    {
      bucket: "storage",
      limit: 3,
      required: false,
      allowedCategories: ["cabinet", "shelf", "tv_stand", "sideboard"],
    },
    {
      bucket: "soft_furnishing",
      limit: 3,
      required: false,
      allowedCategories: ["rug", "window_treatment"],
    },
  ],

  bedroom: [
    {
      bucket: "bed",
      limit: 4,
      required: true,
      allowedCategories: ["bed", "mattress"],
    },
    {
      bucket: "seating",
      limit: 3,
      required: false,
      allowedCategories: ["accent_chair", "bench", "ottoman"],
    },
    {
      bucket: "tables",
      limit: 4,
      required: true,
      allowedCategories: ["side_table"],
    },
    {
      bucket: "lighting",
      limit: 2,
      required: true,
      allowedCategories: ["lamp"],
    },
    {
      bucket: "wall_art",
      limit: 4,
      required: false,
      allowedCategories: ["framed_art", "canvas_art", "map_art", "wall_hanging", "nursery_art"],
    },
    {
      bucket: "decor",
      limit: 3,
      required: false,
      allowedCategories: ["decor", "mirror", "artificial_plant", "floral_arrangement"],
    },
    {
      bucket: "storage",
      limit: 4,
      required: false,
      allowedCategories: ["dresser", "shelf"],
    },
    {
      bucket: "soft_furnishing",
      limit: 5,
      required: false,
      allowedCategories: ["rug", "bedding", "window_treatment"],
    },
  ],

  dining_room: [
    {
      bucket: "tables",
      limit: 4,
      required: true,
      allowedCategories: ["dining_table"],
    },
    {
      bucket: "seating",
      limit: 6,
      required: true,
      allowedCategories: ["dining_chair", "bench"],
    },
    {
      bucket: "lighting",
      limit: 2,
      required: false,
      allowedCategories: ["lamp"],
    },
    {
      bucket: "wall_art",
      limit: 4,
      required: false,
      allowedCategories: [
        "framed_art",
        "canvas_art",
        "canvas_sign",
        "metal_sign",
        "map_art",
        "wall_hanging",
      ],
    },
    {
      bucket: "decor",
      limit: 3,
      required: false,
      allowedCategories: ["decor", "mirror", "artificial_plant", "floral_arrangement"],
    },
    {
      bucket: "storage",
      limit: 3,
      required: false,
      allowedCategories: ["sideboard", "cabinet", "shelf"],
    },
    {
      bucket: "soft_furnishing",
      limit: 2,
      required: false,
      allowedCategories: ["rug", "window_treatment"],
    },
  ],

  kitchen: [
    {
      bucket: "seating",
      limit: 3,
      required: false,
      allowedCategories: ["bar_stool"],
    },
    {
      bucket: "lighting",
      limit: 2,
      required: false,
      allowedCategories: ["lamp"],
    },
    {
      bucket: "wall_art",
      limit: 3,
      required: false,
      allowedCategories: ["canvas_sign", "metal_sign", "framed_art", "canvas_art"],
    },
    {
      bucket: "decor",
      limit: 4,
      required: true,
      allowedCategories: ["decor", "mirror", "artificial_plant", "kitchen"],
    },
    {
      bucket: "storage",
      limit: 6,
      required: false,
      allowedCategories: ["kitchen_storage", "cabinet", "shelf"],
    },
    {
      bucket: "soft_furnishing",
      limit: 2,
      required: false,
      allowedCategories: ["window_treatment"],
    },
  ],

  office: [
    {
      bucket: "seating",
      limit: 4,
      required: true,
      allowedCategories: ["office_chair", "accent_chair", "bench", "ottoman"],
    },
    {
      bucket: "tables",
      limit: 3,
      required: true,
      allowedCategories: ["desk", "side_table"],
    },
    {
      bucket: "lighting",
      limit: 2,
      required: true,
      allowedCategories: ["lamp"],
    },
    {
      bucket: "wall_art",
      limit: 4,
      required: false,
      allowedCategories: ["framed_art", "canvas_art", "canvas_sign", "map_art", "wall_hanging"],
    },
    {
      bucket: "decor",
      limit: 3,
      required: false,
      allowedCategories: ["decor", "mirror", "artificial_plant", "floral_arrangement"],
    },
    {
      bucket: "storage",
      limit: 4,
      required: false,
      allowedCategories: ["shelf", "cabinet"],
    },
    {
      bucket: "soft_furnishing",
      limit: 2,
      required: false,
      allowedCategories: ["rug", "window_treatment"],
    },
  ],
};

const BUCKET_EXCLUDES: Record<string, string[]> = {
  seating: [
    "bedding",
    "bedsheet",
    "sheet",
    "quilt",
    "duvet",
    "comforter",
    "blanket",
    "pillow",
    "sham",
    "slipcover",
    "sofa cover",
    "sink",
    "faucet",
    "playhouse",
  ],
  tables: [
    "bedding",
    "quilt",
    "duvet",
    "comforter",
    "blanket",
    "sink",
    "faucet",
    "playhouse",
    "lamp shade",
    "painting",
  ],
  lighting: [
    "bedding",
    "quilt",
    "duvet",
    "comforter",
    "sink",
    "faucet",
    "playhouse",
    "sofa set",
    "bedroom set",
    "dining set",
  ],
  wall_art: [
    "sink",
    "faucet",
    "playhouse",
    "bedding",
    "quilt",
    "duvet",
    "comforter",
    "pillow",
    "rug",
    "sofa",
    "chair",
    "bed",
    "table",
    "tv stand",
  ],
  decor: [
    "sink",
    "faucet",
    "playhouse",
    "mattress",
  ],
  storage: [
    "sink",
    "faucet",
    "playhouse",
    "bedding",
    "quilt",
    "duvet",
    "comforter",
    "pillow",
  ],
  soft_furnishing: ["sink", "faucet", "playhouse"],
  bed: ["sink", "faucet", "playhouse"],
};

const BUCKET_INCLUDES: Record<string, string[]> = {
  seating: ["sofa", "chair", "armchair", "accent chair", "bench", "ottoman", "stool"],
  tables: ["table", "desk", "nightstand", "side table", "coffee table"],
  lighting: ["lamp", "light", "chandelier", "sconce", "pendant"],
  wall_art: ["art", "canvas", "framed", "print", "poster", "map", "hanging", "sign"],
  decor: ["decor", "vase", "mirror", "plant", "flower", "arrangement", "tray", "candle"],
  storage: ["cabinet", "shelf", "bookcase", "tv stand", "sideboard", "dresser"],
  soft_furnishing: ["rug", "bedding", "curtain", "window", "runner"],
  bed: ["bed", "mattress", "headboard"],
};

function vectorLiteral(values: number[]): string {
  return `[${values.join(",")}]`;
}

function containsAny(text: string, patterns: string[]): boolean {
  const lower = text.toLowerCase();
  return patterns.some((p) => {
    const escaped = p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, "i");
    return regex.test(lower);
  });
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function seededNoise(seed: string, salt: string): number {
  const h = hashString(`${seed}:${salt}`);
  return (h % 1000) / 1000;
}

function rotateArray<T>(items: T[], offset: number): T[] {
  if (!items.length) return items;
  const n = offset % items.length;
  return [...items.slice(n), ...items.slice(0, n)];
}

function rerankBucketItems(
  bucket: string,
  items: RetrievedProduct[],
  theme: string,
  diversificationSeed: string
): RetrievedProduct[] {
  const includes = BUCKET_INCLUDES[bucket] ?? [];
  const excludes = BUCKET_EXCLUDES[bucket] ?? [];
  const themeWords = theme
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((x) => x.trim())
    .filter((x) => x.length > 2);

  const filtered = items.filter((item) => {
    const text = `${item.title} ${item.category ?? ""} ${item.subcategory ?? ""} ${item.normalized_category ?? ""}`.toLowerCase();
    if (containsAny(text, excludes)) return false;
    return true;
  });

  const scored = filtered.map((item) => {
    const text = `${item.title} ${item.category ?? ""} ${item.subcategory ?? ""} ${item.normalized_category ?? ""}`.toLowerCase();

    let score = item.similarity * 10;

    if (containsAny(text, includes)) score += 2.5;

    const overlap = themeWords.filter((w) => text.includes(w)).length;
    score += Math.min(overlap, 4) * 0.35;

    const jitter = seededNoise(diversificationSeed, item.product_handle) * 0.35;
    score += jitter;

    return { ...item, __score: score };
  });

  scored.sort((a, b) => b.__score - a.__score);

  const diversified: typeof scored = [];
  const seenTitleFamilies = new Set<string>();
  const seenHandles = new Set<string>();

  for (const item of scored) {
    const family = item.title.toLowerCase().split(/\s+/).slice(0, 3).join(" ");
    if (seenHandles.has(item.product_handle)) continue;

    if (seenTitleFamilies.has(family) && diversified.length < Math.ceil(scored.length * 0.7)) {
      continue;
    }

    diversified.push(item);
    seenHandles.add(item.product_handle);
    seenTitleFamilies.add(family);
  }

  for (const item of scored) {
    if (!seenHandles.has(item.product_handle)) {
      diversified.push(item);
      seenHandles.add(item.product_handle);
    }
  }

  return diversified.map(({ __score, ...rest }) => rest);
}

async function queryBucketCandidates(
  embedding: number[],
  roomType: RoomType,
  bucket: BucketConfig,
  excludedHandles: string[]
): Promise<RetrievedProduct[]> {
  const vector = vectorLiteral(embedding);
  const candidateLimit = Math.max(bucket.limit * 8, 36);

  const params: unknown[] = [vector, roomType, bucket.bucket, candidateLimit];
  let paramIndex = 5;

  const clauses: string[] = [
    `embedding IS NOT NULL`,
    `primary_image_url IS NOT NULL`,
    `supported_rooms_json ? $2`,
    `retrieval_bucket = $3`,
  ];

  if (bucket.allowedCategories && bucket.allowedCategories.length > 0) {
    clauses.push(`category = ANY($${paramIndex}::text[])`);
    params.push(bucket.allowedCategories);
    paramIndex += 1;
  }

  if (excludedHandles.length > 0) {
    clauses.push(`product_handle <> ALL($${paramIndex}::text[])`);
    params.push(excludedHandles);
    paramIndex += 1;
  }

  const sql = `
    SELECT
      product_handle,
      title,
      category,
      category AS subcategory,
      category AS normalized_category,
      primary_image_url,
      min_price,
      max_price,
      1 - (embedding <=> $1::vector) AS similarity
    FROM products
    WHERE ${clauses.join("\n      AND ")}
    ORDER BY embedding <=> $1::vector
    LIMIT $4
  `;

  const result = await db.query(sql, params);

  return result.rows
    .filter((row: any) => row.primary_image_url)
    .map((row: any) => ({
      bucket: bucket.bucket,
      product_handle: row.product_handle,
      title: row.title,
      category: row.category,
      subcategory: row.subcategory,
      normalized_category: row.normalized_category,
      image_url: row.primary_image_url,
      min_price: row.min_price !== null ? Number(row.min_price) : null,
      max_price: row.max_price !== null ? Number(row.max_price) : null,
      similarity: Number(row.similarity),
    }));
}

function buildDiversificationSeed(roomType: RoomType, theme: string): string {
  const minuteBucket = Math.floor(Date.now() / (1000 * 60));
  return `${roomType}|${theme.toLowerCase()}|${minuteBucket}`;
}

function interleaveByBucket(
  rotatedBuckets: BucketConfig[],
  bucketItemsMap: Record<string, RetrievedProduct[]>,
  pageSize: number
): RetrievedProduct[] {
  const result: RetrievedProduct[] = [];
  const bucketQueues: Record<string, RetrievedProduct[]> = {};

  for (const bucket of rotatedBuckets) {
    bucketQueues[bucket.bucket] = [...(bucketItemsMap[bucket.bucket] ?? [])];
  }

  while (result.length < pageSize) {
    let addedThisRound = false;

    for (const bucket of rotatedBuckets) {
      const queue = bucketQueues[bucket.bucket];
      if (queue && queue.length > 0) {
        result.push(queue.shift()!);
        addedThisRound = true;

        if (result.length >= pageSize) break;
      }
    }

    if (!addedThisRound) break;
  }

  return result;
}

export async function retrieveCatalogue(
  params: RetrieveCatalogueParams
): Promise<RetrievalResult> {
  const {
    roomType,
    theme,
    seenHandles = [],
    rotationCursor = 0,
    pageSize = 18,
  } = params;

  const queryText = `${theme} for ${roomType.replaceAll("_", " ")}`;
  const embedding = await embedQuery(queryText);

  const buckets = ROOM_BUCKETS[roomType];
  const diversificationSeed = buildDiversificationSeed(roomType, theme);

  const bucketResultsMap: Record<string, RetrievedProduct[]> = {};
  const globalExcludedHandles = [...seenHandles];

  for (const bucket of buckets) {
    const candidates = await queryBucketCandidates(
      embedding,
      roomType,
      bucket,
      globalExcludedHandles
    );

    const reranked = rerankBucketItems(
      bucket.bucket,
      candidates,
      theme,
      diversificationSeed
    );

    bucketResultsMap[bucket.bucket] = reranked;
  }

  const rotatedBuckets = rotateArray(
    buckets,
    rotationCursor % Math.max(1, buckets.length)
  );

  const shortlist = interleaveByBucket(rotatedBuckets, bucketResultsMap, pageSize);

  const nextRotationCursor =
    (rotationCursor + 1) % Math.max(1, buckets.length);

  return {
    roomType,
    theme,
    shortlist,
    nextRotationCursor,
  };
}
