from __future__ import annotations

from architecture_comparison_runner import ArchitectureConfig, run_architecture_experiment


def main() -> None:
    config = ArchitectureConfig(
        architecture_name="gin",
        architecture_label="GIN",
    )
    run_architecture_experiment(config)


if __name__ == "__main__":
    main()
