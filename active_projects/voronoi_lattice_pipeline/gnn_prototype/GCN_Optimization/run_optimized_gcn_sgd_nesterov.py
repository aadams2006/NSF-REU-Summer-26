from __future__ import annotations

from gcn_optimization_runner import OptimizedGCNConfig, run_experiment


def main() -> None:
    config = OptimizedGCNConfig(
        optimizer_name="sgd",
        training_strategy="plateau",
        learning_rate=0.01,
        weight_decay=1e-5,
        hidden_dim=24,
        momentum=0.9,
        nesterov=True,
        max_epochs=2000,
        patience=300,
        scheduler_patience=100,
        grad_clip_norm=None,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
