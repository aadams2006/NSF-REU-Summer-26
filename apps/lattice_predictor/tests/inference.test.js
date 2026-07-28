import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";
import {
  InputValidationError,
  buildSampleFromCsv,
  parseCsv,
  predictEnsemble,
} from "../src/inference.js";

const APP_ROOT = new URL("../", import.meta.url);
const REPOSITORY_ROOT = new URL("../../../", import.meta.url);

async function fixture(name) {
  return readFile(new URL(name, APP_ROOT), "utf8");
}

async function referenceInputs() {
  const [nodeText, adjacencyText, bundleText] = await Promise.all([
    fixture("public/examples/node_features.csv"),
    fixture("public/examples/adjacency_area.csv"),
    fixture("public/model_bundle.json"),
  ]);
  return { nodeText, adjacencyText, bundle: JSON.parse(bundleText) };
}

test("browser inference reproduces the saved PyTorch member predictions", async () => {
  const { nodeText, adjacencyText, bundle } = await referenceInputs();
  const sample = buildSampleFromCsv(nodeText, adjacencyText);
  const result = predictEnsemble(sample, bundle);

  assert.equal(sample.stats.nodeCount, 121);
  assert.equal(sample.stats.edgeCount, 288);
  assert.ok(Math.abs(sample.stats.density - 0.03966942148760331) < 1e-12);

  const expectedMembers = new Map([
    [11, 0.032099362],
    [42, 0.032481723],
    [73, 0.032137625],
    [101, 0.032159526],
    [202, 0.03348199],
  ]);
  for (const member of result.memberPredictions) {
    assert.ok(
      Math.abs(member.value - expectedMembers.get(member.seed)) < 2e-7,
      `seed ${member.seed} differs from the saved prediction`,
    );
  }

  assert.ok(Math.abs(result.prediction - 0.032472044) < 2e-7);
  assert.equal(result.distribution.overall, "in");
  assert.ok(result.intervals[95].lower < result.prediction);
  assert.ok(result.intervals[95].upper > result.prediction);
});

test("browser inference matches all 65 saved external-set member predictions", async () => {
  const bundle = JSON.parse(await fixture("public/model_bundle.json"));
  const dataRoot = new URL(
    "active_projects/voronoi_lattice_pipeline/datasets/Lattice_Guess_Prediction_Input_Data/",
    REPOSITORY_ROOT,
  );
  const runRoot = new URL(
    "active_projects/voronoi_lattice_pipeline/gnn_prototype/outputs/" +
      "gcn3_ensemble_uncertainty/run_final_ensemble_uncertainty_v1/per_model/",
    REPOSITORY_ROOT,
  );
  const folders = (await readdir(dataRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("randomness_"))
    .map((entry) => entry.name)
    .sort();

  const expectedBySeed = new Map();
  for (const seed of bundle.model.memberSeeds) {
    const expectedText = await readFile(
      new URL(`member_seed_${seed}/prediction_results.csv`, runRoot),
      "utf8",
    );
    expectedBySeed.set(
      seed,
      parseCsv(expectedText)
        .slice(1)
        .map((row) => Number(row[1])),
    );
  }

  assert.equal(folders.length, 13);
  for (let sampleIndex = 0; sampleIndex < folders.length; sampleIndex += 1) {
    const folderUrl = new URL(`${folders[sampleIndex]}/`, dataRoot);
    const [nodeText, adjacencyText] = await Promise.all([
      readFile(new URL("node_features.csv", folderUrl), "utf8"),
      readFile(new URL("adjacency_area.csv", folderUrl), "utf8"),
    ]);
    const result = predictEnsemble(
      buildSampleFromCsv(nodeText, adjacencyText),
      bundle,
    );

    for (const member of result.memberPredictions) {
      const expected = expectedBySeed.get(member.seed)[sampleIndex];
      assert.ok(
        Math.abs(member.value - expected) < 2e-7,
        `${folders[sampleIndex]}, seed ${member.seed}: ${member.value} != ${expected}`,
      );
    }
  }
});

test("input validation rejects an incompatible node table", async () => {
  const { adjacencyText } = await referenceInputs();
  const invalidNodes = "node_id,z,y\n0,0,0\n1,1,1\n";

  assert.throws(
    () => buildSampleFromCsv(invalidNodes, adjacencyText),
    (error) =>
      error instanceof InputValidationError &&
      error.message.includes('columns named "x" and "y"'),
  );
});

test("coordinate scaling outside the training geometry is flagged", async () => {
  const { nodeText, adjacencyText, bundle } = await referenceInputs();
  const lines = nodeText.trim().split(/\r?\n/);
  const scaledNodes = [
    lines[0],
    ...lines.slice(1).map((line) => {
      const [nodeId, x, y] = line.split(",");
      return `${nodeId},${Number(x) * 2},${Number(y) * 2}`;
    }),
  ].join("\n");

  const sample = buildSampleFromCsv(scaledNodes, adjacencyText);
  const result = predictEnsemble(sample, bundle);

  assert.equal(result.distribution.overall, "out");
  assert.equal(
    result.distribution.checks.find((check) => check.key === "xSpan").status,
    "out",
  );
});
