from __future__ import annotations

from gcn_optimization_runner import OptimizedGCNConfig, run_experiment


def main() -> None:
    config = OptimizedGCNConfig(
        optimizer_name="adamw",
        training_strategy="two_phase",
        weight_decay=1e-5,
        hidden_dim=24,
        lr_phase1=0.003,
        lr_phase2=0.0005,
        epochs_phase1=200,
        epochs_phase2=700,
        patience=999,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
