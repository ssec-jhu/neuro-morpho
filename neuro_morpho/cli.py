from pathlib import Path

import gin

import neuro_morpho.model.base as base
from neuro_morpho.reports import generator


@gin.configurable
def run(
    model: base.BaseModel,
    training_x_dir: str | Path,
    training_y_dir: str | Path,
    testing_x_dir: str | Path,
    testing_y_dir: str | Path,
    model_save_dir: str | Path,
    model_out_y_dir: str | Path,
    stats_output_dir: str | Path,
    report_output_dir: str | Path,
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
    model_out_y_dir = Path(model_out_y_dir)
    model_save_dir = Path(model_save_dir)

    model = model.fit(
        training_x_dir,
        training_y_dir,
        testing_x_dir,
        testing_y_dir,
    )
    model.save(model_save_dir)

    model.predict_dir(testing_x_dir, model_out_y_dir)
    generator.generate_statistics(model_out_y_dir, stats_output_dir)
    generator.generate_statistics(testing_y_dir, stats_output_dir)
    generator.generate_report(stats_output_dir, report_output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="neuro_morpho",
        description="NeuroMorpho training and reporting CLI.",
    )

    parser.add_argument(
        "config",
        type=str,
        default="config.gin",
        help="The path to the gin configuration file.",
    )

    args = parser.parse_args()
    print(f"Loading gin config from {args.config}")
    gin.parse_config_file(args.config)
    run()
