import argparse

import gin
import gin.torch

from neuro_morpho import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="neuro_morpho",
        description="NeuroMorpho training and reporting CLI.",
    )

    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="The path to the gin configuration file.",
    )

    args = parser.parse_args()
    config = args.config or "config.gin"
    print(f"Loading gin config from {config}")
    gin.parse_config_file(config)
    run.run()
