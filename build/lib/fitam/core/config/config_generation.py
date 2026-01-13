from pathlib import Path
from fitam.core.config import EnvConfig, RadialMapConfig, DatasetConfig, EvaluationConfig, ComputeConfig, LoggingConfig, TrainConfig
from fitam.core.common import dump_json_config


def main(output_path: Path):
    # save an env config
    env_config = EnvConfig.RealEnvConfig()

    dump_json_config(env_config, output_path / "env_config.json")

    # save a dataset config
    dataset_config = DatasetConfig.DatasetConfig()
    dump_json_config(dataset_config, output_path / "dataset_config.json")

    # save an evaluation config
    evaluation_config = EvaluationConfig.EvaluationConfig()
    dump_json_config(evaluation_config, output_path / "evaluation_config.json")

    # save a compute config
    compute_config = ComputeConfig.ComputationConfig()
    dump_json_config(compute_config, output_path / "compute_config.json")

    # save a logging config
    logging_config = LoggingConfig.LoggingConfig()
    dump_json_config(logging_config, output_path / "logging_config.json")

    # save an example radial map config
    radial_map_config = RadialMapConfig.RadialMapConfig()
    radial_map_config.farfield_config = RadialMapConfig.FarFieldConfig()
    dump_json_config(radial_map_config, output_path / "radial_map_config.json")
    # save a train config
    train_config = TrainConfig.TrainConfig()
    dump_json_config(train_config, output_path / "train_config.json")


if __name__ == "__main__":
    import argparse
    from fitam import CONFIGS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path,
                        default=CONFIGS_DIR, help="The path to write the config to")
    args = parser.parse_args()

    main(args.output_path)
