from pathlib import Path

import gin

import neuro_morpho.logging.base as log
from neuro_morpho.model import base


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
    validating_x_dir: str | Path,
    validating_y_dir: str | Path,
    testing_x_dir: str | Path,
    testing_y_dir: str | Path,
    model_save_dir: str | Path,
    model_out_y_dir: str | Path,
    model_stats_output_dir: str | Path,
    labeled_stats_output_dir: str | Path,
    report_output_dir: str | Path,
    logger: log.Logger = None,
    train: bool = False,
    get_threshold: bool = False,
    test: bool = False,
    infer: bool = False,
):
    """Run the model on the data and save the results.

    Args:
        model (BaseModel): The model to run
        data_dir (str|Path): The directory containing the data
        output_dir (str|Path): The directory to save the results
    """
    training_x_dir = Path(training_x_dir)
    training_y_dir = Path(training_y_dir)
    validating_x_dir = Path(validating_x_dir)
    validating_y_dir = Path(validating_y_dir)
    testing_x_dir = Path(testing_x_dir)
    testing_y_dir = Path(testing_y_dir)
    model_save_dir = Path(model_save_dir)
    model_out_y_dir = Path(model_out_y_dir)
    model_stats_output_dir = Path(model_stats_output_dir)
    labeled_stats_output_dir = Path(labeled_stats_output_dir)
    report_output_dir = Path(report_output_dir)

    if logger is None:
        raise ValueError("Logger is not provided. Please provide a logger to log the results.")

    model_id = logger.experiment.get_key()

    if train:
        if config := config_str_to_dict(str(gin.config_str(max_line_length=int(1e5)))):
            logger.log_parameters(config)

        logger.log_code(
            folder=Path(__file__).parent,
        )

        model = model.fit(
            training_x_dir,
            training_y_dir,
            validating_x_dir,
            validating_y_dir,
            logger=logger,
            model_id=model_id,
        )

    if get_threshold:  # if there is a need to binarize the output (soft prediction)
        if not train:  # If there was no training, we need to load the model
            checkpoint_dir = model_save_dir / model_id / "checkpoints"
            model.load_checkpoint(checkpoint_dir)
        model_dir = model_save_dir / Path(model_id)
        threshold = model.find_threshold(
            validating_x_dir,
            validating_y_dir,
            model_dir,
        )
    else:
        threshold = None

    """
        Two following options:
        test: Run the model on the test set, consisting of same size images in testing_x_dir
        and its labels in testing_y_dir. The process includes threshold calculation for binarization purposes,
        usiing the validation images in validating_x_dir and their labels in validating_y_dir.
        
        infer: Run the model on the inference set, consisting of images in testing_x_dir. Images could be
        of different size, and the threshold should be provided.
    """
    if test or infer:  # One of them, not both
        if not train:  # If there was no training, we need to load the model
            checkpoint_dir = model_save_dir / model_id / "checkpoints"
            model.load_checkpoint(checkpoint_dir)
        if threshold is None:  # Get the threshold
            model_dir = model_save_dir / model_id
            threshold = model.find_threshold(
                validating_x_dir,
                validating_y_dir,
                model_dir,
            )

        mode = "test" if test else "infer"
        model.predict_dir(
            in_dir=testing_x_dir,
            out_dir=model_out_y_dir,
            threshold=threshold,
            mode=mode,
        )
