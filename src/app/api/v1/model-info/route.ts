import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    model_loaded: false,
    feature_schema: "wrist-relative-63-v1",
    model_version: null,
    message: "No trained classifier is deployed. Collect labeled landmark samples and train the model before enabling predictions.",
  });
}
