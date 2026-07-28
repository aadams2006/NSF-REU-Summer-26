# Lattice GCN Predictor

A public, static web application for the promoted five-member GCN-3 ensemble from
the Summer 2026 NSF REU project. A user supplies one Voronoi lattice graph and
receives:

- the ensemble-mean stiffness prediction in N/mm;
- all five member predictions;
- 80%, 90%, and 95% confidence intervals for the ensemble mean;
- graph statistics and training-range checks; and
- a downloadable JSON result.

Inference runs entirely in the browser. Uploaded research data is neither sent to
a server nor retained.

## Input contract

One sample requires:

1. `node_features.csv` with columns `x` and `y` (an optional `node_id` column is
   accepted); and
2. `adjacency_area.csv`, a symmetric N × N matrix of non-negative edge weights.

The interface accepts the two CSV files directly, a folder containing them, or a
ZIP. It ignores unrelated files from the lattice pipeline.

## Local development

```bash
cd apps/lattice_predictor
npm install
npm run dev
```

Build and test:

```bash
npm test
npm run build
```

## Model bundle

`public/model_bundle.json` contains the five promoted state dictionaries and
their preprocessing scalers in a browser-readable format. Regenerate it from the
checked-in `torch.save` artifacts without installing PyTorch:

```bash
python scripts/export_model_bundle.py
```

The exporter verifies that feature and target scalers are identical across
members and records SHA-256 checksums for provenance. The inference tests compare
all 65 browser-computed member outputs across the 13-sample external set against
the corresponding saved PyTorch predictions.

## Deployment

`.github/workflows/deploy-lattice-predictor.yml` builds this directory and
publishes `dist/` to GitHub Pages after relevant changes reach `main`. The Vite
base path is configured for:

```text
https://aadams2006.github.io/NSF-REU-Summer-26/
```

GitHub Pages must use **GitHub Actions** as its source in the repository
settings.

## Interpretation and limitations

- Held-out test performance: R² 0.9528, RMSE 0.00284, MAE 0.00225.
- External lattice-set performance: R² 0.4024, RMSE 0.00311, MAE 0.00269.
- The displayed confidence intervals quantify disagreement across five model
  initializations. They are not calibrated predictive intervals for true
  stiffness.
- Training-range checks are warnings, not proof that a sample is in
  distribution.
- Predictions are intended for research screening. They do not replace FEA,
  physical testing, or engineering review.
