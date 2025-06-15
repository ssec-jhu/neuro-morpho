from pathlib import Path

import gin

import neuro_morpho.logging.base as log
from neuro_morpho.model import base
from neuro_morpho.reports import generator


def _config_line_filter(line: str) -> bool:
    return len(line) > 0 and not (line.startswith("#") or line.startswith("import"))


def _config_line_to_pair(line: str) -> list[str]:
    return [kv.strip() for kv in line.split("=")]


def config_str_to_dict(config_str: str) -> dict:
    """Converts a Gin.config_str() to a dict for logging with comet.ml"""
    lines = config_str.splitlines()
    return {k: v for k, v in map(_config_line_to_pair, filter(_config_line_filter, lines))}


@gin.configurable
def run(
    model: base.BaseModel,
    training_x_dir: str | Path,
    training_y_dir: str | Path,
    testing_x_dir: str | Path,
    testing_y_dir: str | Path,
    model_save_dir: str | Path,
    model_out_y_dir: str | Path,
    model_stats_output_dir: str | Path,
    labeled_stats_output_dir: str | Path,
    report_output_dir: str | Path,
    logger: log.Logger = None,
    train: bool = False,
    infer: bool = False,
    tile_size: int = 512,
    tile_assembly: str = "nn",
):
    """Run the model on the data and save the results.

    Args:
        model (BaseModel): The model to run
        data_dir (str|Path): The directory containing the data
        output_dir (str|Path): The directory to save the results
    """
    training_x_dir = Path(training_x_dir)
    training_y_dir = Path(training_y_dir)
    testing_x_dir = Path(testing_x_dir)
    testing_y_dir = Path(testing_y_dir)
    model_save_dir = Path(model_save_dir)
    model_out_y_dir = Path(model_out_y_dir)
    model_stats_output_dir = Path(model_stats_output_dir)
    labeled_stats_output_dir = Path(labeled_stats_output_dir)
    report_output_dir = Path(report_output_dir)
    tile_size = tile_size
    tile_assembly = tile_assembly

    if train:
        if logger is not None:
            if config := config_str_to_dict(str(gin.config_str(max_line_length=int(1e5)))):
                logger.log_parameters(config)

            logger.log_code(
                folder=Path(__file__).parent,
            )

            model.exp_id = logger.experiment.get_key()
            model = model.fit(
                training_x_dir,
                training_y_dir,
                testing_x_dir,
                testing_y_dir,
                logger=logger,
            )
        else:
            model = model.fit(
                training_x_dir,
                training_y_dir,
                testing_x_dir,
                testing_y_dir,
            )

        model.save(model_save_dir)

    if infer:
        model.load(model_save_dir / "27b55b978fea46ceb9a072eca9284c7e.pt")
        model.predict_dir(testing_x_dir, model_out_y_dir)

    generator.generate_statistics(model_out_y_dir, model_stats_output_dir)
    generator.generate_statistics(testing_y_dir, labeled_stats_output_dir)
    generator.generate_report(
        model_stats_output_dir,
        labeled_stats_output_dir,
        report_output_dir,
    )
