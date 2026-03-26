import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI, Modality } from "@google/genai";
import { withGeminiRetry } from "@/lib/geminiRetry";

/**
 * place-products/route.ts
 * ========================
 * Step 2 of the new 3-step pipeline.
 *
 * Receives:
 * - emptyRoomImage: the furniture-free room from Step 1 (geometry anchor)
 * - originalRoomImage: original room for context
 * - products: up to 6 catalogue products with their images
 * - theme: style theme (japandi, scandi, etc.)
 * - roomType: living_room, bedroom, etc.
 *
 * Gemini's job is ONLY to fill the empty room with the provided catalogue products.
 * The empty room acts as a perfect anchor — Gemini sees exactly where to place things.
 *
 * This is different from previous approaches:
 * - No geometry generation needed (room shell is already perfect)
 * - No style re-interpretation (empty room is the exact original geometry)
 * - Just fill the gaps with the provided products
 */

const ai = new GoogleGenAI({
  vertexai: true,
  project:  process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || "global",
});

function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}

async function urlToInlineData(url: string): Promise<{ inlineData: { mimeType: string; data: string } } | null> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) {
      console.warn(`urlToInlineData: HTTP ${res.status} for ${url.slice(-60)} — skipping`);
      return null; // Don't retry — caller will try next product in same category
    }
    const buffer = await res.arrayBuffer();
    const contentType = res.headers.get("content-type") || "image/jpeg";
    const mimeType = contentType.split(";")[0].trim();
    return { inlineData: { mimeType, data: Buffer.from(buffer).toString("base64") } };
  } catch (err) {
    console.warn(`urlToInlineData: fetch failed for ${url.slice(-60)} — skipping`);
    return null;
  }
}

function buildPlacementPrompt(
  theme: string,
  roomType: string,
  products: Array<{ title: string; category: string; refIndex: number }>
): string {
  const room = roomType.replace(/_/g, " ");

  // Map each product to its placement instruction with strict spatial rules
  const placements = products.map(({ title, category, refIndex }) => {
    const where: Record<string, string> = {
      sofa:          "place SITTING ON THE FLOOR against the back wall as the main seating piece — legs touching floor",
      accent_chair:  "place SITTING ON THE FLOOR beside the sofa — chair legs on floor, NOT on sofa",
      coffee_table:  "place SITTING ON THE FLOOR in the centre IN FRONT of the sofa — between the sofa and the camera",
      side_table:    "place SITTING ON THE FLOOR beside the sofa arm — small table legs on floor",
      rug:           "lay FLAT on the FLOOR under the sofa and coffee table — a 2D flat textile on floor, no height",
      lamp:          "place STANDING ON THE FLOOR in the corner of the room beside the sofa — tall standing lamp with base ON THE FLOOR, NEVER placed on sofa or table surface",
      ottoman:       "place SITTING ON THE FLOOR directly in front of the sofa — low seat on floor",
      shelf:         "place STANDING ON THE FLOOR against the side wall — upright unit with base on floor",
      cabinet:       "place STANDING ON THE FLOOR against the wall — storage unit base on floor",
      dining_table:  "place STANDING ON THE FLOOR in the centre of the room — table legs on floor",
      dining_chair:  "place SITTING ON THE FLOOR around the dining table — chair legs on floor",
      bed:           "place SITTING ON THE FLOOR against the main wall — bed frame on floor",
      dresser:       "place STANDING ON THE FLOOR against the wall — drawer unit base on floor",
      mirror:        "hang FLAT AGAINST THE WALL surface — mounted on wall",
    };
    const instruction = where[category] || "place on the floor in an appropriate position";
    return `Image ${refIndex + 2}: "${title}" (${category}) — ${instruction}`;
  }).join("\n");

  return `You are a professional interior designer placing furniture into an empty room.

IMAGE 1: Empty ${room} — this is the EXACT room geometry you must preserve completely.
         The walls, floor, windows, ceiling, camera angle — all must remain exactly as shown.
         This room has just had its furniture removed. Your job is to fill it back in.

${products.length > 0 ? `IMAGES 2-${products.length + 1}: Catalogue furniture products to place into the room:
${placements}` : ""}

YOUR TASK:
1. Keep Image 1's room geometry EXACTLY — same walls, floor colour, window positions, ceiling, camera angle
2. Place each catalogue product naturally into the empty room with correct perspective and scale
3. Add realistic shadows and lighting that matches the room's existing light sources
4. The result must look like a professional interior design photo
5. Style: ${theme}

CRITICAL RULES:
- Do NOT change the wall colour, floor, windows, or ceiling
- Do NOT change the camera angle or room proportions
- Place furniture at realistic scale relative to the room
- The products must look like they actually belong in this room
- ALL furniture must sit on the FLOOR — nothing floats in the air or sits on top of other furniture
- LAMPS stand on the FLOOR beside the sofa — never on the sofa cushions or on a table unless it is specifically a table lamp
- RUGS lie FLAT on the FLOOR — they have no height, they are flat textile
- Output ONLY the furnished room image — nothing else

This is a product placement task, not a room generation task.`;
}

export async function POST(req: NextRequest) {
  try {
    const body              = await req.json();
    const emptyRoomImage    = body?.emptyRoomImage;
    const originalRoomImage = body?.originalRoomImage;
    const products          = body?.products || [];
    const theme             = body?.theme || "modern";
    const roomType          = body?.roomType || "living_room";
    const mimeType          = body?.mimeType || "image/jpeg";

    if (!emptyRoomImage) {
      return NextResponse.json({ ok: false, error: "emptyRoomImage is required" }, { status: 400 });
    }
    if (!products.length) {
      return NextResponse.json({ ok: false, error: "products array is required" }, { status: 400 });
    }

    // Fetch product images
    console.log(`Fetching ${products.length} product images for placement...`);
    const productParts: Array<{ inlineData: { mimeType: string; data: string } }> = [];
    const successfulProducts: Array<{ title: string; category: string; refIndex: number }> = [];

    for (let i = 0; i < Math.min(products.length, 5); i++) {
      const product = products[i];
      if (!product.imageUrl) continue;
      const part = await urlToInlineData(product.imageUrl);
      if (part) {
        productParts.push(part);
        successfulProducts.push({
          title:    product.title,
          category: product.category,
          refIndex: i,
        });
        console.log(`  ✅ [${product.category}] ${product.title.slice(0, 50)}`);
      } else {
        console.warn(`  ⚠️  Failed to fetch: ${product.title.slice(0, 50)}`);
      }
    }

    if (!successfulProducts.length) {
      return NextResponse.json({ ok: false, error: "No product images could be fetched" }, { status: 500 });
    }

    const prompt = buildPlacementPrompt(theme, roomType, successfulProducts);
    const emptyRoomMimeType = emptyRoomImage.startsWith("data:") 
      ? emptyRoomImage.split(";")[0].split(":")[1]
      : "image/png";

    // Build parts:
    // Part 1: prompt text
    // Part 2: empty room image (geometry anchor)
    // Parts 3+: catalogue product images
    const parts: any[] = [
      { text: prompt },
      { text: "Image 1 — EMPTY ROOM (geometry anchor — preserve exactly):" },
      {
        inlineData: {
          mimeType: emptyRoomMimeType,
          data: stripDataUrlPrefix(emptyRoomImage),
        },
      },
      { text: `Images 2-${successfulProducts.length + 1} — CATALOGUE PRODUCTS to place:` },
      ...productParts,
    ];

    console.log("Placing catalogue products into empty room:", {
      theme,
      roomType,
      products: successfulProducts.map(p => `[${p.category}] ${p.title.slice(0, 40)}`),
    });

    const response = await withGeminiRetry(() =>
      ai.models.generateContent({
        model:    "gemini-2.5-flash-image",
        contents: [{ role: "user", parts }],
        config: {
          responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
      })
    );

    // Extract generated image
    const candidates = response?.candidates || [];
    let generatedImageData: string | null = null;
    let generatedMimeType = "image/png";
    let responseText = "";

    for (const candidate of candidates) {
      for (const part of candidate?.content?.parts || []) {
        if (part?.inlineData?.data) {
          generatedImageData = part.inlineData.data;
          generatedMimeType  = part.inlineData.mimeType || "image/png";
        }
        if (typeof part?.text === "string") {
          responseText += part.text;
        }
      }
    }

    if (!generatedImageData) {
      console.warn("Gemini placement returned no image. Text:", responseText.slice(0, 200));
      return NextResponse.json({
        ok:    false,
        error: "Gemini did not return an image for product placement",
        text:  responseText.slice(0, 200),
      }, { status: 500 });
    }

    console.log("Product placement succeeded");

    return NextResponse.json({
      ok:             true,
      generatedImage: `data:${generatedMimeType};base64,${generatedImageData}`,
      productsPlaced: successfulProducts.map(p => ({ title: p.title, category: p.category })),
      productCount:   successfulProducts.length,
    });

  } catch (error) {
    console.error("place-products error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to place products" },
      { status: 500 }
    );
  }
}
