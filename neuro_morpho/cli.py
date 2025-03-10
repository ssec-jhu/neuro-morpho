import argparse

import gin

from neuro_morpho import run

if __name__ == "__main__":
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
    run.run()
