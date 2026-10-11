import { NextResponse } from "next/server";
import { loadClassifier } from "@/lib/asl-classifier";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const model = loadClassifier();
  return NextResponse.json({
    status: "ok", service: "asl-bridge-inference", model_loaded: Boolean(model),
    model_version: model?.model_version ?? null, feature_schema: "wrist-relative-63-v1",
    message: model ? "API and classifier are ready." : "API is online; no valid trained classifier artifact is deployed.",
  });
}
