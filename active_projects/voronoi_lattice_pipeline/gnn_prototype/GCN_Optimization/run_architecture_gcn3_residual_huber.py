from __future__ import annotations

from architecture_comparison_runner import ArchitectureConfig, run_architecture_experiment


def main() -> None:
    config = ArchitectureConfig(
        architecture_name="gcn3_residual",
        architecture_label="GCN-3 Residual Huber",
        hidden_dim=24,
        dropout=0.10,
        lr_phase1=0.0025,
        lr_phase2=0.00045,
        weight_decay=1e-5,
        loss_name="huber",
        huber_beta=0.75,
        output_group="gcn3_residual_huber",
    )
    run_architecture_experiment(config)


if __name__ == "__main__":
    main()
