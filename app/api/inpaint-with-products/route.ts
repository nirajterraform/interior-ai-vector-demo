import { NextRequest, NextResponse } from "next/server";
import { GoogleAuth } from "google-auth-library";
import sharp from "sharp";

/**
 * inpaint-with-products/route.ts
 * ================================
 * TEST ROUTE — uses Imagen 3 subject reference customisation to place
 * actual catalogue product images into the room.
 *
 * API: imagen-3.0-capability-001 with REFERENCE_TYPE_SUBJECT
 * This is the "Imagen for Editing and Customization" model in Model Garden.
 *
 * Flow:
 *   1. Receive room photo + up to 3 catalogue product images (sofa, table, rug)
 *   2. Build a subject reference request with product images as subjects
 *   3. Imagen places those specific products into the room scene
 *   4. Return the generated image
 *
 * Request body:
 * {
 *   imageBase64: string,          // original room photo
 *   mimeType?: string,
 *   theme: string,                // e.g. "japandi living room"
 *   roomType: string,             // e.g. "living_room"
 *   products: [                   // catalogue products to place (max 3)
 *     {
 *       imageUrl: string,         // catalogue product image URL
 *       title: string,            // product title
 *       category: string,         // e.g. "sofa", "coffee_table", "rug"
 *     }
 *   ]
 * }
 */

const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT!;
const LOCATION   = "us-central1";
const MODEL      = "imagen-3.0-capability-001";

let authClient: GoogleAuth | null = null;

function getAuthClient(): GoogleAuth {
  if (!authClient) {
    authClient = new GoogleAuth({
      keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
      scopes: ["https://www.googleapis.com/auth/cloud-platform"],
    });
  }
  return authClient;
}

function stripDataUrlPrefix(input: string): string {
  const idx = input.indexOf(",");
  return idx >= 0 ? input.slice(idx + 1) : input;
}

async function urlToBase64(url: string): Promise<{ data: string; mimeType: string }> {
  const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
  if (!res.ok) throw new Error(`Failed to fetch product image: ${url} (${res.status})`);
  const buffer = await res.arrayBuffer();
  const contentType = res.headers.get("content-type") || "image/jpeg";
  const mimeType = contentType.split(";")[0].trim();
  return {
    data: Buffer.from(buffer).toString("base64"),
    mimeType,
  };
}

async function resizeImage(
  imageBase64: string,
  maxSize = 1024
): Promise<{ data: string; width: number; height: number }> {
  const buffer = Buffer.from(imageBase64, "base64");
  const meta   = await sharp(buffer).metadata();
  const w = meta.width  || 768;
  const h = meta.height || 768;
  const scale = Math.min(maxSize / w, maxSize / h, 1);
  const nw = Math.floor((w * scale) / 8) * 8;
  const nh = Math.floor((h * scale) / 8) * 8;

  const resized = await sharp(buffer)
    .resize(nw, nh)
    .jpeg({ quality: 90 })
    .toBuffer();

  return { data: resized.toString("base64"), width: nw, height: nh };
}

function buildPromptWithSubjectRefs(
  theme: string,
  roomType: string,
  products: Array<{ title: string; category: string; refId: number }>
): string {
  const room = roomType.replace(/_/g, " ");

  // Build product placement instructions referencing each subject by [$refId]
  const placements = products.map(({ title, category, refId }) => {
    const where: Record<string, string> = {
      sofa:           "as the main seating against the wall",
      accent_chair:   "as an accent chair beside the sofa",
      coffee_table:   "in front of the sofa as the coffee table",
      side_table:     "beside the sofa as a side table",
      rug:            "on the floor under the furniture",
      lamp:           "beside the seating as a floor lamp",
      ottoman:        "in front of the sofa as a footrest",
      shelf:          "against the wall as a bookshelf",
      cabinet:        "against the wall for storage",
      dining_table:   "as the central dining table",
      dining_chair:   "around the dining table",
      bed:            "as the main bed against the wall",
      dresser:        "against the wall as a dresser",
      mirror:         "on the wall above the sofa or sideboard",
    };
    const placement = where[category] || "in an appropriate location in the room";
    return `place the [$${refId}] (${title}) ${placement}`;
  }).join(", ");

  return [
    `${theme} style ${room} interior design`,
    placements,
    "photorealistic interior photography, natural lighting, 8k resolution",
    "preserve the room walls, floor, windows, ceiling exactly as they are",
    "only replace the furniture with the referenced products",
  ].join(", ");
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const imageBase64Input = body?.imageBase64;
    const mimeType         = body?.mimeType || "image/jpeg";
    const theme            = body?.theme || "modern interior";
    const roomType         = body?.roomType || "living_room";
    const products         = body?.products || [];

    if (!imageBase64Input) {
      return NextResponse.json({ ok: false, error: "imageBase64 is required" }, { status: 400 });
    }
    if (!Array.isArray(products) || products.length === 0) {
      return NextResponse.json({ ok: false, error: "products array is required (min 1)" }, { status: 400 });
    }
    // Imagen allows max 2 total reference images for non-square aspect ratios.
    // Since 1 slot is used by the room photo (REFERENCE_TYPE_RAW),
    // we can only use 1 product reference for landscape rooms.
    // For square images, up to 3 product references are allowed.
    if (products.length > 3) {
      return NextResponse.json({ ok: false, error: "max 3 products supported" }, { status: 400 });
    }
    if (!PROJECT_ID) {
      return NextResponse.json({ ok: false, error: "GOOGLE_CLOUD_PROJECT not configured" }, { status: 500 });
    }

    // ── Resize room image ──────────────────────────────────────────────────
    const rawBase64 = stripDataUrlPrefix(imageBase64Input);
    let { data: roomImage, width, height } = await resizeImage(rawBase64);

    // Check if image is square — Imagen allows max 2 total refs for non-square
    // (1 room + 1 product). For square images, up to 4 refs allowed.
    const isSquare = Math.abs(width - height) < 32;
    const maxProducts = isSquare ? 3 : 1;
    const effectiveProducts = products.slice(0, maxProducts);

    if (!isSquare && products.length > 1) {
      console.log(`Non-square image (${width}x${height}) — limiting to 1 product reference (Imagen API constraint)`);
    }

    // ── Fetch and resize product images ───────────────────────────────────
    console.log(`Fetching ${effectiveProducts.length} product images for subject reference inpainting...`);
    const productImages: Array<{
      data: string;
      mimeType: string;
      title: string;
      category: string;
      refId: number;
    }> = [];

    for (let i = 0; i < effectiveProducts.length; i++) {
      const product = effectiveProducts[i];
      try {
        const { data, mimeType: pMime } = await urlToBase64(product.imageUrl);
        // Resize product image to max 512x512 for reference
        const resized = await resizeImage(data, 512);
        productImages.push({
          data:     resized.data,
          mimeType: pMime,
          title:    product.title,
          category: product.category,
          refId:    i + 1,
        });
        console.log(`  ✅ Fetched product ${i + 1}: ${product.title.slice(0, 50)}`);
      } catch (err) {
        console.warn(`  ⚠️  Failed to fetch product ${i + 1} (${product.title}):`, err);
        // Skip this product — don't fail the whole request
      }
    }

    if (productImages.length === 0) {
      return NextResponse.json(
        { ok: false, error: "All product images failed to load" },
        { status: 500 }
      );
    }

    // ── Build prompt with subject references ──────────────────────────────
    const prompt = buildPromptWithSubjectRefs(theme, roomType, productImages);
    console.log("Subject reference prompt:", prompt.slice(0, 200));

    // ── Get access token ───────────────────────────────────────────────────
    const auth        = getAuthClient();
    const accessToken = await auth.getAccessToken();
    if (!accessToken) throw new Error("Failed to get Google Cloud access token");

    // ── Build Imagen customisation request ────────────────────────────────
    // REFERENCE_TYPE_RAW  = the base room image to edit
    // REFERENCE_TYPE_SUBJECT = the product to place in the scene (up to 3)
    const referenceImages: any[] = [
      {
        referenceType: "REFERENCE_TYPE_RAW",
        referenceId:   0,
        referenceImage: { bytesBase64Encoded: roomImage },
      },
      ...productImages.map((p) => ({
        referenceType: "REFERENCE_TYPE_SUBJECT",
        referenceId:   p.refId,
        referenceImage: { bytesBase64Encoded: p.data },
        subjectImageConfig: {
          subjectType:        "SUBJECT_TYPE_DEFAULT",
          subjectDescription: `${p.category} furniture: ${p.title}`,
        },
      })),
    ];

    const requestBody = {
      instances: [{
        prompt,
        referenceImages,
      }],
      parameters: {
        sampleCount: 1,
        seed: Math.floor(Math.random() * 1000000),
      },
    };

    const endpoint = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL}:predict`;

    console.log("Calling Imagen subject reference customisation...", {
      width, height,
      products: productImages.map((p) => `[$${p.refId}] ${p.title.slice(0, 40)}`),
    });

    const response = await fetch(endpoint, {
      method:  "POST",
      headers: {
        Authorization:  `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Imagen customisation API error:", response.status, errText.slice(0, 500));
      throw new Error(`Imagen API error ${response.status}: ${errText.slice(0, 400)}`);
    }

    const result = await response.json();

    if (result?.error) {
      throw new Error(`Imagen returned error: ${JSON.stringify(result.error)}`);
    }

    const predictions = result?.predictions;
    if (!predictions?.length) {
      throw new Error("Imagen returned no predictions");
    }

    const generatedBase64 = predictions[0]?.bytesBase64Encoded;
    if (!generatedBase64) {
      throw new Error("Imagen prediction did not contain an image");
    }

    return NextResponse.json({
      ok: true,
      generatedImage:   `data:image/png;base64,${generatedBase64}`,
      width,
      height,
      prompt,
      productsUsed:     productImages.map((p) => ({ title: p.title, category: p.category, refId: p.refId })),
      productCount:     productImages.length,
    });

  } catch (error) {
    console.error("inpaint-with-products error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to inpaint with products" },
      { status: 500 }
    );
  }
}
