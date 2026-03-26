import { NextRequest, NextResponse } from "next/server";
import { GoogleAuth } from "google-auth-library";
import sharp from "sharp";

/**
 * remove-furniture/route.ts
 * ==========================
 * Step 1 of the new 3-step pipeline.
 *
 * Uses Imagen 3 EDIT_MODE_INPAINT_REMOVAL with MASK_MODE_FOREGROUND
 * to remove ALL furniture from the room, leaving the exact room shell:
 * - Walls preserved pixel-perfect
 * - Floor preserved pixel-perfect
 * - Windows preserved pixel-perfect
 * - Ceiling preserved pixel-perfect
 * - Only furniture/foreground objects removed → replaced with empty space
 *
 * This gives us the room geometry anchor for Step 2 (Gemini product placement).
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

async function resizeForImagen(imageBase64: string): Promise<{
  data: string;
  width: number;
  height: number;
}> {
  const buffer = Buffer.from(imageBase64, "base64");
  const meta   = await sharp(buffer).metadata();
  const w = meta.width  || 768;
  const h = meta.height || 768;
  const scale = Math.min(1024 / w, 1024 / h, 1);
  const nw = Math.floor((w * scale) / 8) * 8;
  const nh = Math.floor((h * scale) / 8) * 8;

  const resized = await sharp(buffer)
    .resize(nw, nh)
    .jpeg({ quality: 90 })
    .toBuffer();

  return { data: resized.toString("base64"), width: nw, height: nh };
}

export async function POST(req: NextRequest) {
  try {
    const body           = await req.json();
    const imageBase64Raw = body?.imageBase64;
    const roomType       = body?.roomType || "living_room";

    if (!imageBase64Raw) {
      return NextResponse.json({ ok: false, error: "imageBase64 is required" }, { status: 400 });
    }
    if (!PROJECT_ID) {
      return NextResponse.json({ ok: false, error: "GOOGLE_CLOUD_PROJECT not configured" }, { status: 500 });
    }

    const rawBase64 = stripDataUrlPrefix(imageBase64Raw);
    const { data: roomImage, width, height } = await resizeForImagen(rawBase64);

    const auth        = getAuthClient();
    const accessToken = await auth.getAccessToken();
    if (!accessToken) throw new Error("Failed to get Google Cloud access token");

    // EDIT_MODE_INPAINT_REMOVAL + MASK_MODE_FOREGROUND:
    // Imagen automatically detects all foreground objects (furniture, lamps, rugs etc.)
    // and removes them, filling the space naturally (floor/wall continuation)
    // Prompt describes what the room looks like AFTER removal — helps Imagen
    // fill the empty areas correctly
    const room = roomType.replace(/_/g, " ");
    const removalPrompt = `empty ${room} with bare floor and walls, no furniture, clean interior`;

    const requestBody = {
      instances: [{
        prompt: removalPrompt,
        referenceImages: [
          {
            referenceType:  "REFERENCE_TYPE_RAW",
            referenceId:    1,
            referenceImage: { bytesBase64Encoded: roomImage },
          },
          {
            referenceType:  "REFERENCE_TYPE_MASK",
            referenceId:    2,
            // No referenceImage — use automatic foreground detection
            maskImageConfig: {
              maskMode: "MASK_MODE_FOREGROUND",
              dilation: 0.02,  // small dilation to catch furniture edges
            },
          },
        ],
      }],
      parameters: {
        editMode:   "EDIT_MODE_INPAINT_REMOVAL",
        editConfig: { baseSteps: 20 },  // removal needs fewer steps than insertion
        sampleCount: 1,
        seed: 42,  // fixed seed for consistency
      },
    };

    const endpoint = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL}:predict`;

    console.log("Removing furniture from room:", { width, height, roomType });

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
      console.error("Imagen removal API error:", response.status, errText.slice(0, 400));
      throw new Error(`Imagen API error ${response.status}: ${errText.slice(0, 300)}`);
    }

    const result = await response.json();

    if (result?.error) {
      throw new Error(`Imagen error: ${JSON.stringify(result.error)}`);
    }

    const predictions = result?.predictions;
    if (!predictions?.length || !predictions[0]?.bytesBase64Encoded) {
      throw new Error("Imagen removal returned no image");
    }

    const emptyRoomBase64 = predictions[0].bytesBase64Encoded;

    console.log("Furniture removed successfully — empty room ready for product placement");

    return NextResponse.json({
      ok:             true,
      emptyRoomImage: `data:image/png;base64,${emptyRoomBase64}`,
      originalImage:  imageBase64Raw,  // pass through for reference
      width,
      height,
    });

  } catch (error) {
    console.error("remove-furniture error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to remove furniture" },
      { status: 500 }
    );
  }
}
