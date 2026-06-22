from __future__ import annotations

from gcn_optimization_runner import OptimizedGCNConfig, run_experiment


def main() -> None:
    config = OptimizedGCNConfig(
        optimizer_name="adamw",
        learning_rate=0.0015,
        weight_decay=1e-4,
        max_epochs=600,
        patience=60,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
