import { NextResponse } from "next/server";
import { loadClassifier, predictClassifier } from "@/lib/asl-classifier";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ status: "invalid_request", prediction: null, detail: "Expected a JSON body." }, { status: 400 });
  }

  const features = (body as { features?: unknown } | null)?.features;
  if (!Array.isArray(features) || features.length !== 63 ||
      !features.every((value) => typeof value === "number" && Number.isFinite(value))) {
    return NextResponse.json({
      status: "invalid_request",
      prediction: null,
      detail: "features must contain exactly 63 finite numbers.",
    }, { status: 422 });
  }

  const model = loadClassifier();
  if (!model) {
    return NextResponse.json({
      status: "model_unavailable",
      prediction: null,
      detail: "API is online, but models/sign_classifier.json is missing or invalid. Train and export a real model first.",
    }, { status: 503 });
  }

  try {
    const prediction = predictClassifier(model, features);
    const status = prediction.confidence < 0.55 ? "uncertain" : "recognized";
    return NextResponse.json({
      status,
      prediction,
      model_version: model.model_version,
    });
  } catch {
    return NextResponse.json({
      status: "inference_error",
      prediction: null,
      detail: "Classifier inference failed.",
    }, { status: 500 });
  }
}
