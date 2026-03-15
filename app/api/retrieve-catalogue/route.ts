import { NextRequest, NextResponse } from "next/server";
import { retrieveCatalogue } from "@/lib/retrieval";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const roomType = body?.roomType;
    const theme = body?.theme;
    const seenHandles = Array.isArray(body?.seenHandles) ? body.seenHandles : [];
    const rotationCursor =
      typeof body?.rotationCursor === "number" ? body.rotationCursor : 0;
    const pageSize =
      typeof body?.pageSize === "number" && body.pageSize > 0
        ? body.pageSize
        : 18;

    if (!roomType) {
      return NextResponse.json(
        { error: "roomType is required" },
        { status: 400 }
      );
    }

    if (!theme || !String(theme).trim()) {
      return NextResponse.json(
        { error: "theme is required" },
        { status: 400 }
      );
    }

    const result = await retrieveCatalogue({
      roomType,
      theme,
      seenHandles,
      rotationCursor,
      pageSize,
    });

    return NextResponse.json(result);
  } catch (error) {
    console.error("retrieve-catalogue error:", error);

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to retrieve catalogue",
      },
      { status: 500 }
    );
  }
}
