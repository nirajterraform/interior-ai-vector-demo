import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI, Modality } from "@google/genai";
import { withGeminiRetry } from "@/lib/geminiRetry";

/**
 * fill-room/route.ts
 * ==================
 * Step 2 of the pipeline.
 *
 * Receives:
 *   - emptyRoomImage : cleaned room from /api/clean-room (no furniture)
 *   - products       : catalogue products from vector search
 *   - theme          : user selected theme e.g. "japandi living room..."
 *   - roomType       : "living_room" etc.
 *
 * Sends to Gemini:
 *   Image 1 : empty room  → geometry anchor, do not change anything
 *   Images 2+: catalogue product photos → place these exact items
 *
 * Gemini's ONLY job: place the products into the empty room.
 * Do not change walls, floor, ceiling, windows, colour, or camera angle.
 */

const ai = new GoogleGenAI({
  vertexai: true,
  project:  process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || "global",
});

function stripDataUrlPrefix(b64: string): string {
  const i = b64.indexOf(",");
  return i >= 0 ? b64.slice(i + 1) : b64;
}

async function fetchProductImage(
  url: string
): Promise<{ inlineData: { mimeType: string; data: string } } | null> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) return null;
    const buf  = await res.arrayBuffer();
    const mime = (res.headers.get("content-type") || "image/jpeg").split(";")[0].trim();
    return { inlineData: { mimeType: mime, data: Buffer.from(buf).toString("base64") } };
  } catch {
    return null;
  }
}

function buildPrompt(
  roomType: string,
  theme: string,
  products: Array<{ title: string; category: string; index: number }>
): string {
  const room = roomType.replace(/_/g, " ");

  const POSITION: Record<string, string> = {
    sofa:         "on the floor against the back wall — the main seating piece",
    coffee_table: "on the floor in the centre of the room, directly in front of the sofa",
    rug:          "flat on the floor, underneath the sofa and coffee table",
    lamp:         "on the floor in the corner of the room beside the sofa — a tall standing lamp",
    accent_chair: "on the floor beside the sofa — a single seat",
    ottoman:      "on the floor directly in front of the sofa",
    side_table:   "on the floor beside the sofa arm — a small table",
    shelf:        "on the floor against the side wall — an upright storage unit",
    cabinet:      "on the floor against the wall — a storage cabinet",
  };

  const productLines = products
    .map(({ title, category, index }) => {
      const pos = POSITION[category] || "on the floor in a natural position";
      return `  Image ${index + 2} = "${title}" (${category}) → place ${pos}`;
    })
    .join("\n");

  return `You are an interior design tool. Your job is to replace the existing furniture in a room with exact catalogue products.

━━━ IMAGE 1: ORIGINAL ROOM ━━━
This is the user's room with their existing furniture.
- The walls, floor, windows, ceiling, camera angle are ALL FIXED — do not change them
- Note where the existing sofa, coffee table, rug, and lamp are positioned
- You will replace these pieces with the catalogue products shown in Images 2+

━━━ CATALOGUE PRODUCTS TO PLACE (Images 2 onwards) ━━━
Replace the existing furniture with these exact products:
${productLines}

━━━ WHAT YOU MUST DO ━━━
1. Remove all existing movable furniture (sofa, chairs, tables, rugs, lamps, ottomans, baskets)
2. Place each catalogue product in the same position as the original furniture it replaces
3. Match each product's exact appearance from its reference image — colour, shape, material
4. Each piece sits on the floor with correct perspective and realistic shadows
5. The result looks like a professional interior photograph
6. Apply style: ${theme}

━━━ WHAT YOU MUST NOT DO ━━━
- Do NOT change walls, floor, windows, ceiling, curtains, camera angle
- Do NOT use backgrounds from product photos — ignore studio backdrops
- Do NOT invent furniture not shown in the catalogue product images
- Do NOT change the colour or style of any catalogue product
- Do NOT place items on top of each other

━━━ SELF-CHECK BEFORE OUTPUTTING ━━━
✓ Walls, floor, windows identical to Image 1?
✓ Every piece matches its catalogue product image?
✓ No invented objects added?
✓ Nothing floating or stacked?

Output: one photorealistic furnished room image only.`.trim();
}

export async function POST(req: NextRequest) {
  try {
    const body         = await req.json();
    const emptyRoom    = body?.emptyRoomImage  as string;
    const products     = body?.products        as Array<{ imageUrl: string; title: string; category: string }>;
    const theme        = body?.theme           as string || "modern";
    const roomType     = body?.roomType        as string || "living_room";

    if (!emptyRoom)       return NextResponse.json({ ok: false, error: "emptyRoomImage required" }, { status: 400 });
    if (!products?.length) return NextResponse.json({ ok: false, error: "products required" }, { status: 400 });

    // Fetch product images — try up to 3 per slot, skip broken URLs
    const fetched: Array<{ inlineData: { mimeType: string; data: string } }> = [];
    const placed:  Array<{ title: string; category: string; index: number }> = [];

    for (const product of products.slice(0, 5)) {
      if (!product.imageUrl) continue;
      const img = await fetchProductImage(product.imageUrl);
      if (img) {
        placed.push({ title: product.title, category: product.category, index: fetched.length });
        fetched.push(img);
        console.log(`  ✅ [${product.category}] ${product.title.slice(0, 50)}`);
      } else {
        console.warn(`  ⚠️  Skipped (bad URL): ${product.title.slice(0, 50)}`);
      }
    }

    if (!fetched.length) {
      return NextResponse.json({ ok: false, error: "No product images could be fetched" }, { status: 500 });
    }

    const emptyRoomMime = emptyRoom.startsWith("data:")
      ? emptyRoom.split(";")[0].split(":")[1]
      : "image/jpeg";

    const parts: any[] = [
      { text: buildPrompt(roomType, theme, placed) },
      { text: "Image 1 — ORIGINAL ROOM (study the existing furniture positions, then replace them with the catalogue products — keep all walls, floor, windows, ceiling exactly as shown):" },
      { inlineData: { mimeType: emptyRoomMime, data: stripDataUrlPrefix(emptyRoom) } },
      { text: `Images 2–${fetched.length + 1} — CATALOGUE PRODUCTS (place these into the room):` },
      ...fetched,
    ];

    console.log(`Filling room with ${placed.length} catalogue products:`, {
      theme: theme.slice(0, 60),
      products: placed.map(p => `[${p.category}] ${p.title.slice(0, 40)}`),
    });

    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model:    "gemini-2.5-flash-image",
        contents: [{ role: "user", parts }],
        config:   { responseModalities: [Modality.IMAGE, Modality.TEXT] },
      })
    );

    // Extract image from response
    let imageData: string | null = null;
    let imageMime = "image/png";
    let responseText = "";

    for (const candidate of response?.candidates || []) {
      for (const part of candidate?.content?.parts || []) {
        if (part?.inlineData?.data) {
          imageData = part.inlineData.data;
          imageMime = part.inlineData.mimeType || "image/png";
        }
        if (typeof part?.text === "string") responseText += part.text;
      }
    }

    if (!imageData) {
      console.warn("Gemini returned no image. Text:", responseText.slice(0, 200));
      return NextResponse.json({
        ok: false,
        error: "Gemini did not return a room image",
        text: responseText.slice(0, 200),
      }, { status: 500 });
    }

    console.log("Room filled successfully with catalogue products");

    return NextResponse.json({
      ok:            true,
      generatedImage: `data:${imageMime};base64,${imageData}`,
      productsPlaced: placed.map(p => ({ title: p.title, category: p.category })),
    });

  } catch (error) {
    console.error("fill-room error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to fill room" },
      { status: 500 }
    );
  }
}
