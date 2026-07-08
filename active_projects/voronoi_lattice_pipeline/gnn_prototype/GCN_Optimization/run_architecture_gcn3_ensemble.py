from __future__ import annotations

from ensemble_runner import EnsembleConfig, run_fixed_split_ensemble


def main() -> None:
    config = EnsembleConfig(
        architecture_name="gcn3",
        architecture_label="GCN-3",
        member_seeds=(11, 42, 73, 101, 202),
        split_seed=42,
        hidden_dim=24,
        dropout=0.10,
        lr_phase1=0.003,
        lr_phase2=0.0005,
        weight_decay=1e-5,
        output_group="gcn3_ensemble_fixed_split",
    )
    run_fixed_split_ensemble(config)


if __name__ == "__main__":
    main()
