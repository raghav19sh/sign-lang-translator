import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    service: "asl-bridge-inference",
    model_loaded: false,
    feature_schema: "wrist-relative-63-v1",
    message: "API is online; a trained classifier has not been deployed.",
  });
}
