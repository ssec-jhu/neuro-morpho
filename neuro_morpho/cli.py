from pathlib import Path

import neuro_morpho.model.base as base
from neuro_morpho.reports import generator


def run(
    model: base.BaseModel,
    training_dir: str | Path,
    testing_dir: str | Path,
    model_save_dir: str | Path,
    test_output_dir: str | Path,
    report_output_dir: str | Path,
):
    """Run the model on the data and save the results.

    Args:
        model (BaseModel): The model to run
        data_dir (str|Path): The directory containing the data
        output_dir (str|Path): The directory to save the results
    """

    training_dir = Path(training_dir)
    testing_dir = Path(testing_dir)
    model_save_dir = Path(model_save_dir)

    model = model.fit(training_dir)
    model.save(model_save_dir)

    model.predict_dir(testing_dir, test_output_dir)
    generator.generate_report(test_output_dir, testing_dir, report_output_dir)
    generator.generate_plots(report_output_dir)
