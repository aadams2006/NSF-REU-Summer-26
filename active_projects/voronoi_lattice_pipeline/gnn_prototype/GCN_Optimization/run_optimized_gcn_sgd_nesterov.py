from __future__ import annotations

from gcn_optimization_runner import OptimizedGCNConfig, run_experiment


def main() -> None:
    config = OptimizedGCNConfig(
        optimizer_name="sgd",
        learning_rate=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        nesterov=True,
        max_epochs=600,
        patience=80,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
