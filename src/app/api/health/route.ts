import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    service: "asl-bridge",
    recognition: "requires-trained-model",
    modelLoaded: false,
    featureSchema: "wrist-relative-63-v1",
  });
}
