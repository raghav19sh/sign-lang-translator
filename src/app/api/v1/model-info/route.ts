import { NextResponse } from "next/server";
import { loadClassifier } from "@/lib/asl-classifier";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const model = loadClassifier();
  return NextResponse.json({
    model_loaded: Boolean(model), feature_schema: "wrist-relative-63-v1",
    model_version: model?.model_version ?? null, classes: model?.classes ?? [],
    tree_count: model?.trees.length ?? 0, metrics: model?.metrics ?? null,
    message: model ? "Trained classifier loaded." : "No trained classifier is deployed. Collect labeled samples, train, export, and deploy the JSON artifact.",
  });
}
