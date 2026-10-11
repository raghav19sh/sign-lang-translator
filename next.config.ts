import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  outputFileTracingIncludes: {
    "/api/v1/health": ["./models/sign_classifier.json"],
    "/api/v1/model-info": ["./models/sign_classifier.json"],
    "/api/v1/recognize": ["./models/sign_classifier.json"],
  },
};

export default nextConfig;
