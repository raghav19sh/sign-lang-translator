import { NextResponse } from "next/server";

export const runtime = "nodejs";

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

  return NextResponse.json({
    status: "model_unavailable",
    prediction: null,
    detail: "The API is online, but no trained classifier artifact is deployed. No prediction was fabricated.",
  }, { status: 503 });
}
