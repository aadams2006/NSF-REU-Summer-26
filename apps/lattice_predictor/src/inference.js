const MAX_NODE_COUNT = 1_000;
const SYMMETRY_TOLERANCE = 1e-5;
const T_CRITICAL_DF4 = {
  80: 1.5332062740589432,
  90: 2.131846786326649,
  95: 2.7764451051977987,
};

const f32 = Math.fround;

export class InputValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "InputValidationError";
  }
}

export function parseCsv(text) {
  const source = String(text).replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
      continue;
    }

    if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field.trim());
      field = "";
    } else if (character === "\n") {
      row.push(field.trim());
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else if (character !== "\r") {
      field += character;
    }
  }

  row.push(field.trim());
  if (row.some((value) => value !== "")) rows.push(row);

  if (quoted) {
    throw new InputValidationError("A CSV field has an unmatched quote.");
  }
  return rows;
}

function finiteNumber(value, description) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new InputValidationError(`${description} must be a finite number.`);
  }
  return parsed;
}

export function parseNodeFeaturesCsv(text) {
  const rows = parseCsv(text);
  if (rows.length < 3) {
    throw new InputValidationError("node_features.csv must contain a header and at least two nodes.");
  }

  const header = rows[0].map((value) => value.trim().toLowerCase());
  const xIndex = header.indexOf("x");
  const yIndex = header.indexOf("y");
  if (xIndex < 0 || yIndex < 0) {
    throw new InputValidationError('node_features.csv must include columns named "x" and "y".');
  }

  const coordinates = rows.slice(1).map((row, rowIndex) => {
    if (row.length <= Math.max(xIndex, yIndex)) {
      throw new InputValidationError(`Node row ${rowIndex + 2} is missing an x or y value.`);
    }
    return [
      f32(finiteNumber(row[xIndex], `x at node row ${rowIndex + 2}`)),
      f32(finiteNumber(row[yIndex], `y at node row ${rowIndex + 2}`)),
    ];
  });

  if (coordinates.length > MAX_NODE_COUNT) {
    throw new InputValidationError(
      `This browser predictor accepts at most ${MAX_NODE_COUNT.toLocaleString()} nodes per sample.`,
    );
  }

  return coordinates;
}

export function parseAdjacencyCsv(text, nodeCount) {
  let rows = parseCsv(text);

  if (rows.length === nodeCount + 1) {
    rows = rows.slice(1);
  }
  if (rows.length !== nodeCount) {
    throw new InputValidationError(
      `adjacency_area.csv has ${rows.length} data rows; ${nodeCount} are required to match the node table.`,
    );
  }

  const matrix = rows.map((row, rowIndex) => {
    let values = row;
    if (values.length === nodeCount + 1) values = values.slice(1);
    if (values.length !== nodeCount) {
      throw new InputValidationError(
        `Adjacency row ${rowIndex + 1} has ${values.length} values; ${nodeCount} are required.`,
      );
    }
    return Float32Array.from(
      values.map((value, columnIndex) => {
        const parsed = finiteNumber(
          value,
          `adjacency value at row ${rowIndex + 1}, column ${columnIndex + 1}`,
        );
        if (parsed < 0) {
          throw new InputValidationError("Adjacency weights must be zero or positive.");
        }
        return parsed;
      }),
    );
  });

  let edgeCount = 0;
  for (let row = 0; row < nodeCount; row += 1) {
    if (Math.abs(matrix[row][row]) > SYMMETRY_TOLERANCE) {
      throw new InputValidationError("The adjacency matrix diagonal must contain zeros.");
    }
    for (let column = row + 1; column < nodeCount; column += 1) {
      const forward = matrix[row][column];
      const reverse = matrix[column][row];
      const tolerance =
        SYMMETRY_TOLERANCE * Math.max(1, Math.abs(forward), Math.abs(reverse));
      if (Math.abs(forward - reverse) > tolerance) {
        throw new InputValidationError(
          `The adjacency matrix must be symmetric (mismatch at nodes ${row} and ${column}).`,
        );
      }
      if (forward > 0) edgeCount += 1;
    }
  }

  if (edgeCount === 0) {
    throw new InputValidationError("The adjacency matrix does not contain any edges.");
  }
  return matrix;
}

function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function populationStd(values, valuesMean = mean(values)) {
  if (values.length === 0) return 0;
  return Math.sqrt(
    values.reduce((total, value) => total + (value - valuesMean) ** 2, 0) / values.length,
  );
}

function closeTo(value, reference) {
  return Math.abs(value - reference) <= 1e-8 + 1e-5 * Math.abs(reference);
}

export function deriveGraphRepresentation(nodeCoordinates, adjacency) {
  const nodeCount = nodeCoordinates.length;
  const xValues = nodeCoordinates.map(([x]) => x);
  const yValues = nodeCoordinates.map(([, y]) => y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const centerX = f32(mean(xValues));
  const centerY = f32(mean(yValues));

  const degree = new Float32Array(nodeCount);
  const weightedDegree = new Float32Array(nodeCount);
  const incidentDistanceSum = new Float32Array(nodeCount);
  const maxIncidentDistance = new Float32Array(nodeCount);
  const edges = [];
  const edgeWeights = [];

  for (let row = 0; row < nodeCount; row += 1) {
    for (let column = row + 1; column < nodeCount; column += 1) {
      const weight = adjacency[row][column];
      if (weight <= 0) continue;

      const dx = f32(nodeCoordinates[row][0] - nodeCoordinates[column][0]);
      const dy = f32(nodeCoordinates[row][1] - nodeCoordinates[column][1]);
      const distance = f32(Math.sqrt(f32(dx * dx + dy * dy)));

      degree[row] = f32(degree[row] + 1);
      degree[column] = f32(degree[column] + 1);
      weightedDegree[row] = f32(weightedDegree[row] + weight);
      weightedDegree[column] = f32(weightedDegree[column] + weight);
      incidentDistanceSum[row] = f32(incidentDistanceSum[row] + distance);
      incidentDistanceSum[column] = f32(incidentDistanceSum[column] + distance);
      maxIncidentDistance[row] = Math.max(maxIncidentDistance[row], distance);
      maxIncidentDistance[column] = Math.max(maxIncidentDistance[column], distance);

      edges.push([row, column, weight]);
      edgeWeights.push(weight);
    }
  }

  const centerDistances = nodeCoordinates.map(([x, y]) => {
    const dx = f32(x - centerX);
    const dy = f32(y - centerY);
    return f32(Math.sqrt(f32(dx * dx + dy * dy)));
  });

  const nodeFeatures = nodeCoordinates.map(([x, y], index) => {
    const safeDegree = Math.max(1, degree[index]);
    const boundary = Number(
      closeTo(x, minX) || closeTo(x, maxX) || closeTo(y, minY) || closeTo(y, maxY),
    );
    return Float32Array.of(
      f32(x),
      f32(y),
      degree[index],
      weightedDegree[index],
      centerDistances[index],
      f32(boundary),
      f32(incidentDistanceSum[index] / safeDegree),
      maxIncidentDistance[index],
      f32(weightedDegree[index] / safeDegree),
    );
  });

  const edgeCount = edges.length;
  const density =
    nodeCount <= 1 ? 0 : (2 * edgeCount) / (nodeCount * (nodeCount - 1));
  const degreeValues = Array.from(degree);
  const weightedDegreeValues = Array.from(weightedDegree);
  const degreeMean = mean(degreeValues);
  const weightedDegreeMean = mean(weightedDegreeValues);
  const edgeWeightMean = mean(edgeWeights);
  const centerDistanceMean = mean(centerDistances);

  const graphFeatures = Float32Array.of(
    f32(nodeCount),
    f32(edgeCount),
    f32(density),
    f32(edgeWeights.reduce((total, value) => total + value, 0)),
    f32(edgeWeightMean),
    f32(populationStd(edgeWeights, edgeWeightMean)),
    f32(degreeMean),
    f32(populationStd(degreeValues, degreeMean)),
    f32(weightedDegreeMean),
    f32(populationStd(weightedDegreeValues, weightedDegreeMean)),
    f32(centerDistanceMean),
    f32(populationStd(centerDistances, centerDistanceMean)),
  );

  return {
    nodeCoordinates,
    adjacency,
    nodeFeatures,
    graphFeatures,
    edges,
    stats: {
      nodeCount,
      edgeCount,
      density,
      meanDegree: degreeMean,
      stdDegree: populationStd(degreeValues, degreeMean),
      meanWeightedDegree: weightedDegreeMean,
      meanEdgeWeight: edgeWeightMean,
      xSpan: maxX - minX,
      ySpan: maxY - minY,
    },
  };
}

export function buildSampleFromCsv(nodeFeaturesText, adjacencyText) {
  const nodeCoordinates = parseNodeFeaturesCsv(nodeFeaturesText);
  const adjacency = parseAdjacencyCsv(adjacencyText, nodeCoordinates.length);
  return deriveGraphRepresentation(nodeCoordinates, adjacency);
}

function scaleRows(rows, scaler) {
  if (rows.length === 0 || rows[0].length !== scaler.mean.length) {
    throw new Error("The model bundle and derived node features have incompatible dimensions.");
  }
  return rows.map((row) =>
    Float32Array.from(
      row,
      (value, index) =>
        f32((value - scaler.mean[index]) / (scaler.scale[index] || 1)),
    ),
  );
}

function scaleVector(values, scaler) {
  if (values.length !== scaler.mean.length) {
    throw new Error("The model bundle and derived graph features have incompatible dimensions.");
  }
  return Float32Array.from(
    values,
    (value, index) => f32((value - scaler.mean[index]) / (scaler.scale[index] || 1)),
  );
}

function linearRows(rows, weights) {
  return rows.map((row) => {
    const output = new Float32Array(weights.length);
    for (let outputIndex = 0; outputIndex < weights.length; outputIndex += 1) {
      const weightRow = weights[outputIndex];
      let total = 0;
      for (let inputIndex = 0; inputIndex < row.length; inputIndex += 1) {
        total += row[inputIndex] * weightRow[inputIndex];
      }
      output[outputIndex] = f32(total);
    }
    return output;
  });
}

function gcnLayer(rows, adjacency, weights, bias) {
  const transformed = linearRows(rows, weights);
  const nodeCount = rows.length;
  const outputDimension = weights.length;
  const degrees = new Float32Array(nodeCount);

  for (let row = 0; row < nodeCount; row += 1) {
    let degree = 1;
    for (let column = 0; column < nodeCount; column += 1) {
      if (row !== column && adjacency[row][column] > 0) {
        degree += adjacency[row][column];
      }
    }
    degrees[row] = f32(degree);
  }

  const output = Array.from({ length: nodeCount }, () => new Float32Array(outputDimension));
  for (let target = 0; target < nodeCount; target += 1) {
    const selfWeight = 1 / degrees[target];
    for (let feature = 0; feature < outputDimension; feature += 1) {
      let total = selfWeight * transformed[target][feature];
      for (let source = 0; source < nodeCount; source += 1) {
        if (source === target) continue;
        const edgeWeight = adjacency[target][source];
        if (edgeWeight <= 0) continue;
        const normalization = edgeWeight / Math.sqrt(degrees[target] * degrees[source]);
        total += normalization * transformed[source][feature];
      }
      output[target][feature] = f32(total + bias[feature]);
    }
  }
  return output;
}

function batchNormalize(rows, state, prefix) {
  const gamma = state[`${prefix}.weight`];
  const beta = state[`${prefix}.bias`];
  const runningMean = state[`${prefix}.running_mean`];
  const runningVariance = state[`${prefix}.running_var`];
  const epsilon = 1e-5;

  return rows.map((row) =>
    Float32Array.from(row, (value, index) =>
      f32(
        ((value - runningMean[index]) / Math.sqrt(runningVariance[index] + epsilon)) *
          gamma[index] +
          beta[index],
      ),
    ),
  );
}

function reluRows(rows) {
  return rows.map((row) => Float32Array.from(row, (value) => Math.max(0, value)));
}

function globalMeanPool(rows) {
  const output = new Float32Array(rows[0].length);
  for (const row of rows) {
    for (let index = 0; index < row.length; index += 1) {
      output[index] = f32(output[index] + row[index]);
    }
  }
  for (let index = 0; index < output.length; index += 1) {
    output[index] = f32(output[index] / rows.length);
  }
  return output;
}

function predictMember(scaledNodes, scaledGraph, adjacency, member, targetScaler) {
  const state = member.state;

  let hidden = gcnLayer(
    scaledNodes,
    adjacency,
    state["conv1.lin.weight"],
    state["conv1.bias"],
  );
  hidden = reluRows(batchNormalize(hidden, state, "bn1"));

  hidden = gcnLayer(
    hidden,
    adjacency,
    state["conv2.lin.weight"],
    state["conv2.bias"],
  );
  hidden = reluRows(batchNormalize(hidden, state, "bn2"));

  hidden = gcnLayer(
    hidden,
    adjacency,
    state["conv3.lin.weight"],
    state["conv3.bias"],
  );
  hidden = reluRows(batchNormalize(hidden, state, "bn3"));

  const pooled = globalMeanPool(hidden);
  const combined = Float32Array.from([...pooled, ...scaledGraph]);
  const outputWeights = state["output.weight"][0];
  let standardizedPrediction = state["output.bias"][0];
  for (let index = 0; index < combined.length; index += 1) {
    standardizedPrediction += combined[index] * outputWeights[index];
  }

  return f32(f32(standardizedPrediction) * targetScaler.scale + targetScaler.mean);
}

function classifyReferenceCheck(value, range) {
  const numericalTolerance =
    Math.max(1, Math.abs(range.min), Math.abs(range.max)) * 1e-6;
  if (
    value >= range.min - numericalTolerance &&
    value <= range.max + numericalTolerance
  ) {
    return "in";
  }
  const span = range.max - range.min;
  const midpoint = (range.min + range.max) / 2;
  const tolerance = Math.max(span * 0.1, Math.abs(midpoint) * 0.01, 1e-6);
  if (value >= range.min - tolerance && value <= range.max + tolerance) return "near";
  return "out";
}

export function evaluateDistribution(stats, reference) {
  const definitions = [
    ["nodeCount", "Node count", stats.nodeCount],
    ["edgeCount", "Edge count", stats.edgeCount],
    ["density", "Graph density", stats.density],
    ["meanDegree", "Mean degree", stats.meanDegree],
    ["stdDegree", "Degree spread", stats.stdDegree],
    ["meanWeightedDegree", "Mean weighted degree", stats.meanWeightedDegree],
    ["meanEdgeWeight", "Mean edge weight", stats.meanEdgeWeight],
    ["xSpan", "X span", stats.xSpan],
    ["ySpan", "Y span", stats.ySpan],
  ];

  const checks = definitions.map(([key, label, value]) => {
    const range = reference[key];
    return {
      key,
      label,
      value,
      min: range.min,
      max: range.max,
      status: classifyReferenceCheck(value, range),
    };
  });

  const overall = checks.some((check) => check.status === "out")
    ? "out"
    : checks.some((check) => check.status === "near")
      ? "near"
      : "in";
  return { overall, checks };
}

export function predictEnsemble(sample, bundle) {
  if (!bundle?.members?.length) throw new Error("The model bundle does not contain ensemble members.");

  const scaledNodes = scaleRows(
    sample.nodeFeatures,
    bundle.preprocessing.nodeFeatureScaler,
  );
  const scaledGraph = scaleVector(
    sample.graphFeatures,
    bundle.preprocessing.graphFeatureScaler,
  );
  const memberPredictions = bundle.members.map((member) => ({
    seed: member.seed,
    value: predictMember(
      scaledNodes,
      scaledGraph,
      sample.adjacency,
      member,
      bundle.preprocessing.targetScaler,
    ),
  }));

  const values = memberPredictions.map(({ value }) => value);
  const ensembleMean = f32(mean(values));
  const sampleVariance =
    values.reduce((total, value) => total + (value - ensembleMean) ** 2, 0) /
    Math.max(1, values.length - 1);
  const memberStd = Math.sqrt(sampleVariance);
  const standardError = memberStd / Math.sqrt(values.length);
  const intervals = Object.fromEntries(
    Object.entries(T_CRITICAL_DF4).map(([level, critical]) => {
      const halfWidth = critical * standardError;
      return [
        level,
        {
          lower: ensembleMean - halfWidth,
          upper: ensembleMean + halfWidth,
          halfWidth,
        },
      ];
    }),
  );

  return {
    prediction: ensembleMean,
    units: bundle.model.units,
    memberPredictions,
    memberStd,
    memberMin: Math.min(...values),
    memberMax: Math.max(...values),
    memberRange: Math.max(...values) - Math.min(...values),
    relativeMemberStdPercent:
      Math.abs(ensembleMean) > Number.EPSILON
        ? (memberStd / Math.abs(ensembleMean)) * 100
        : null,
    intervals,
    distribution: evaluateDistribution(
      sample.stats,
      bundle.preprocessing.trainingReference,
    ),
    graph: sample.stats,
    model: {
      name: bundle.model.name,
      run: bundle.model.run,
      memberCount: bundle.members.length,
      splitSeed: bundle.model.splitSeed,
    },
  };
}
