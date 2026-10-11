import { readFileSync } from "node:fs";
import path from "node:path";

type Tree = {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  values: number[][];
};
export type PortableForest = {
  model_version: string;
  feature_schema: "wrist-relative-63-v1";
  classes: string[];
  trees: Tree[];
  metrics?: { accuracy?: number; macro_f1?: number; samples_total?: number; class_count?: number; split_method?: string };
};

let cached: PortableForest | null | undefined;

export function loadClassifier(): PortableForest | null {
  if (cached !== undefined) return cached;
  try {
    const artifactPath = path.join(process.cwd(), "models", "sign_classifier.json");
    const parsed = JSON.parse(readFileSync(artifactPath, "utf8")) as PortableForest;
    if (parsed.feature_schema !== "wrist-relative-63-v1" ||
        !Array.isArray(parsed.classes) || parsed.classes.length < 2 ||
        !Array.isArray(parsed.trees) || parsed.trees.length === 0 ||
        parsed.trees.some((tree) => !Array.isArray(tree.children_left) ||
          !Array.isArray(tree.children_right) || !Array.isArray(tree.feature) ||
          !Array.isArray(tree.threshold) || !Array.isArray(tree.values))) {
      cached = null;
      return cached;
    }
    cached = parsed;
  } catch {
    cached = null;
  }
  return cached;
}

export function predictClassifier(model: PortableForest, features: number[]) {
  const totals = new Array<number>(model.classes.length).fill(0);
  for (const tree of model.trees) {
    let node = 0;
    let steps = 0;
    while (tree.children_left[node] !== tree.children_right[node] && steps <= tree.children_left.length) {
      const featureIndex = tree.feature[node];
      if (featureIndex < 0 || featureIndex >= features.length) break;
      node = features[featureIndex] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
      steps += 1;
    }
    const counts = tree.values[node] ?? [];
    const sum = counts.reduce((total, count) => total + count, 0);
    if (sum > 0) {
      for (let i = 0; i < totals.length; i += 1) totals[i] += (counts[i] ?? 0) / sum;
    }
  }
  const denominator = model.trees.length || 1;
  const probabilities = totals.map((value) => value / denominator);
  let best = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[best]) best = i;
  }
  return { label: model.classes[best], confidence: probabilities[best] ?? 0 };
}
