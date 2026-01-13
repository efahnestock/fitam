from pathlib import Path
from fitam import CONFIGS_DIR, SWATHS_DIR, EVALUATION_REQUESTS_DIR, MODELS_DIR
from fitam.pipeline.pipeline_elements import Evaluation
from fitam.core.common import create_dir, dump_json_config


def save_all_inputs(eval: Evaluation) -> None:
    def process_path(path: Path | None, rel_path: Path) -> str | None:
        if path is not None:
            if type(path) == tuple:
                print(path)
            return str(path.relative_to(rel_path)) if path.is_relative_to(rel_path) else str(path)
        else:
            return None
    all_inputs = dict(
        name=eval.name,
        dataset_config_path=process_path(eval.dataset_config_path, CONFIGS_DIR),
        training_config_path=process_path(eval.training_config_path, CONFIGS_DIR),
        evaluation_config_path=process_path(eval.evaluation_config_path, CONFIGS_DIR),
        compute_config_path=process_path(eval.compute_config_path, CONFIGS_DIR),
        radial_map_config_path=process_path(eval.radial_map_config_path, CONFIGS_DIR),
        swath_library_path=process_path(eval.swath_library_path, SWATHS_DIR),
        logging_config_path=process_path(eval.logging_config_path, CONFIGS_DIR),
        eval_request_paths=[process_path(p, EVALUATION_REQUESTS_DIR) for p in eval.eval_request_paths],
        save_paths=[str(x) for x in eval.save_paths],
        num_active_bins=eval.num_active_bins,
    )
    if eval.model_path is not None:
        all_inputs['model_path'] = str(Path(eval.model_path).relative_to(MODELS_DIR)) if Path(eval.model_path).is_relative_to(MODELS_DIR) else str(eval.model_path)
    create_dir(eval.save_root_path)
    if not (eval.save_root_path / "inputs.json").exists():
        dump_json_config(all_inputs, eval.save_root_path / "inputs.json")
    else:
        print("inputs.json already exists, overwriting")
        dump_json_config(all_inputs, eval.save_root_path / "inputs.json", overwrite=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    " ASSUMES ALL OF THE PATHS ARE COMPLETE i.e. not relative to __DIR "
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('-d', '--dataset_config_path', type=str, default=None)
    parser.add_argument('-t', '--training_config_path', type=str, default=None)
    parser.add_argument('-e', '--evaluation_config_path', type=str, default=None)
    parser.add_argument('-c', '--compute_config_path', type=str, default=None)
    parser.add_argument('-r', '--radial_map_config_path', type=str, default=None)
    parser.add_argument('-s', '--swath_library_path', type=str, default=None)
    parser.add_argument('-l', '--logging_config_path', type=str, default=None)
    parser.add_argument('-q', '--eval_request_paths', type=str, nargs="+", default=None)
    parser.add_argument('-p', '--save_paths', type=str, nargs="+", default=None)
    parser.add_argument('-m', '--model_path', type=str, required=False)
    parser.add_argument('-n', '--num_active_bins', type=int, default=None)
    parser.add_argument('-o', '--save_root_path', type=str, default=None)

    args = parser.parse_args()

    eval = Evaluation(
        name=args.name,
        dataset_config_path=Path(args.dataset_config_path) if args.dataset_config_path is not None else None,
        training_config_path=Path(args.training_config_path) if args.training_config_path is not None else None,
        evaluation_config_path=Path(args.evaluation_config_path) if args.evaluation_config_path is not None else None,
        compute_config_path=Path(args.compute_config_path) if args.compute_config_path is not None else None,
        radial_map_config_path=Path(args.radial_map_config_path) if args.radial_map_config_path is not None else None,
        swath_library_path=Path(args.swath_library_path) if args.swath_library_path is not None else None,
        logging_config_path=Path(args.logging_config_path) if args.logging_config_path is not None else None,
        eval_request_paths=[Path(p) for p in args.eval_request_paths],
        save_paths=[Path(p) for p in args.save_paths],
        model_path=Path(args.model_path) if args.model_path is not None else None,
        num_active_bins=args.num_active_bins,
        save_root_path=Path(args.save_root_path)
    )
    save_all_inputs(eval)
    print(f"Saved inputs to {eval.save_root_path / 'inputs.json'}")
