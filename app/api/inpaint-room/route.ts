import { NextRequest, NextResponse } from "next/server";
import { GoogleAuth } from "google-auth-library";
import sharp from "sharp";

// ── Imagen 3 inpainting via Vertex AI REST API ────────────────────────────────
// Model: imagen-3.0-capability-001
// Supports: EDIT_MODE_INPAINT_INSERTION with MASK_MODE_FOREGROUND (auto-detects furniture)
// Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/image/edit-insert-objects

const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT!;
const LOCATION = "us-central1"; // Imagen requires us-central1, not global
const MODEL = "imagen-3.0-capability-001";

// Auth client — reuse across requests
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

// Furniture type anchors per room — tells Imagen what TYPE of furniture
// to keep so it doesn't freely swap a sofa for accent chairs
// For REMOVAL mode: prompt describes the result after furniture is removed
// Keep minimal — just describe the empty room floor/walls that should be revealed
// The catalogue furniture will be placed by multi-blend in the next step
function buildInpaintPrompt(_theme: string, roomType: string): string {
  const room = roomType.replace(/_/g, " ");
  return `empty ${room}, bare floor, bare walls, no furniture, no objects, photorealistic interior photograph`;
}

function buildNegativePrompt(): string {
  return [
    "ugly, blurry, distorted, deformed, low quality",
    "watermark, text, signature, cartoon, painting, illustration",
    "changed walls, different floor, different windows, different room geometry",
    "remove sofa, remove couch, replace sofa with chairs, no sofa",
    "completely different furniture layout, different room configuration",
  ].join(", ");
}

async function resizeForImagen(imageBase64: string, mimeType: string): Promise<{ data: string; width: number; height: number }> {
  // Imagen 3 requires dimensions to be multiples of 8
  // Max recommended: 1024x1024 for inpainting
  const buffer = Buffer.from(imageBase64, "base64");
  const metadata = await sharp(buffer).metadata();
  const origW = metadata.width || 768;
  const origH = metadata.height || 768;

  // Scale to fit within 1024x1024 keeping aspect ratio, snapping to multiples of 8
  const scale = Math.min(1024 / origW, 1024 / origH, 1);
  const newW = Math.floor((origW * scale) / 8) * 8;
  const newH = Math.floor((origH * scale) / 8) * 8;

  const resized = await sharp(buffer)
    .resize(newW, newH)
    .jpeg({ quality: 90 })
    .toBuffer();

  return {
    data: resized.toString("base64"),
    width: newW,
    height: newH,
  };
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const imageBase64Input = body?.imageBase64;
    const mimeType = body?.mimeType || "image/jpeg";
    const theme = body?.theme || "modern interior design";
    const roomType = body?.roomType || "living_room";
    // Optional: user-provided mask. If not provided, we use automatic foreground detection
    const maskBase64Input = body?.maskBase64 || null;

    if (!imageBase64Input) {
      return NextResponse.json({ ok: false, error: "imageBase64 is required" }, { status: 400 });
    }
    if (!PROJECT_ID) {
      return NextResponse.json({ ok: false, error: "GOOGLE_CLOUD_PROJECT not configured" }, { status: 500 });
    }

    // Resize image to Imagen-compatible dimensions
    const imageBase64Raw = stripDataUrlPrefix(imageBase64Input);
    const { data: resizedImage, width, height } = await resizeForImagen(imageBase64Raw, mimeType);

    // Get access token
    const auth = getAuthClient();
    const accessToken = await auth.getAccessToken();
    if (!accessToken) {
      throw new Error("Failed to get Google Cloud access token");
    }

    const prompt = buildInpaintPrompt(theme, roomType);
    const negativePrompt = buildNegativePrompt();

    // Build the Vertex AI Imagen request
    // Using EDIT_MODE_INPAINT_REMOVAL with MASK_MODE_FOREGROUND:
    // Removes foreground furniture while preserving background geometry
    // Result: same room with furniture neutralised, ready for catalogue blend
    // This tells Imagen to: detect furniture (foreground) → replace with new content → keep background pixel-perfect
    let maskConfig: any;
    if (maskBase64Input) {
      // Use provided mask
      const maskBase64Raw = stripDataUrlPrefix(maskBase64Input);
      maskConfig = {
        referenceType: "REFERENCE_TYPE_MASK",
        referenceId: 2,
        referenceImage: { bytesBase64Encoded: maskBase64Raw },
        maskImageConfig: {
          maskMode: "MASK_MODE_USER_PROVIDED",
          dilation: 0.03, // small dilation to smooth edges
        },
      };
    } else {
      // Auto-detect foreground (furniture) — referenceImage must be OMITTED entirely
      // (not null) when using MASK_MODE_FOREGROUND, otherwise Imagen returns null_value error
      maskConfig = {
        referenceType: "REFERENCE_TYPE_MASK",
        referenceId: 2,
        maskImageConfig: {
          maskMode: "MASK_MODE_FOREGROUND",
          dilation: 0.03,
        },
      };
    }

    const requestBody = {
      instances: [
        {
          prompt,
          negativePrompt,
          referenceImages: [
            {
              referenceType: "REFERENCE_TYPE_RAW",
              referenceId: 1,
              referenceImage: { bytesBase64Encoded: resizedImage },
            },
            maskConfig,
          ],
        },
      ],
      parameters: {
        editMode: "EDIT_MODE_INPAINT_REMOVAL",
        editConfig: {
          baseSteps: 20,
        },
        sampleCount: 1,
        seed: Math.floor(Math.random() * 1000000),
      },
    };

    const endpoint = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL}:predict`;

    console.log("Imagen REMOVAL — preserving geometry:", { width, height, prompt: prompt.slice(0, 80) });

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Imagen API error:", response.status, errText);
      throw new Error(`Imagen API error ${response.status}: ${errText.slice(0, 500)}`);
    }

    const result = await response.json();
    const predictions = result?.predictions;

    if (!predictions?.length) {
      throw new Error("Imagen returned no predictions");
    }

    // Extract the generated image
    const generatedBase64 = predictions[0]?.bytesBase64Encoded;
    if (!generatedBase64) {
      throw new Error("Imagen prediction did not contain an image");
    }

    const mimeTypeOut = "image/png";
    const generatedImage = `data:${mimeTypeOut};base64,${generatedBase64}`;

    return NextResponse.json({
      ok: true,
      generatedImage,
      width,
      height,
      prompt,
    });

  } catch (error) {
    console.error("inpaint-room error:", error);
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Failed to inpaint room",
      },
      { status: 500 }
    );
  }
}
