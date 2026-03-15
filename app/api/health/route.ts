import { NextResponse } from "next/server";
import { testDbConnection } from "@/lib/db";

export async function GET() {
  try {
    const dbInfo = await testDbConnection();

    return NextResponse.json({
      ok: true,
      service: "interior-ai-vector-demo",
      database: {
        connected: true,
        name: dbInfo.current_database,
        user: dbInfo.user_name,
        time: dbInfo.now,
        productsCount: Number(dbInfo.products_count),
      },
    });
  } catch (error) {
    console.error("Health check failed:", error);

    return NextResponse.json(
      {
        ok: false,
        service: "interior-ai-vector-demo",
        database: {
          connected: false,
        },
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}