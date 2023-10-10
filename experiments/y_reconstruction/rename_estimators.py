import argparse
from pathlib import Path
import pandas as pd


def rename_estimators(input_path: Path):
    print(f"Renaming estimators from {input_path}...")
    data = pd.read_table(input_path)
    output_path = input_path.with_stem(input_path.stem + "_renamed")

    wrapper_renaming = {
        "regressor_to_classifier": "",
        "nrlmf_y_reconstruction": "__nrlmf",
        "dnilmf_y_reconstruction": "__dnilmf",
    }

    # Rename estimators
    data["suffix"] = (
        data["wrapper.name"].map(wrapper_renaming).fillna(data["wrapper.name"])
    )
    data["estimator.name"] += data["suffix"]
    data.to_csv(output_path, sep="\t", index=False)

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename estimators in a runs table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to the runs table to rename.",
    )
    args = parser.parse_args()

    rename_estimators(args.results)


if __name__ == "__main__":
    main()
