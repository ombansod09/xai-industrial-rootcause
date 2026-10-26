import argparse
from dataclasses import asdict

from xai_industrial.config import load_config
from xai_industrial.eda.profile import run_eda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--timestamp-column")
    args = parser.parse_args()

    config = load_config(args.config)
    timestamp_column = args.timestamp_column or config.data.timestamp_column
    artifacts = run_eda(
        args.input,
        metadata_path=args.metadata,
        output_dir=args.output,
        timestamp_column=timestamp_column,
        config=asdict(config),
    )
    print(artifacts.summary_json)
    print(artifacts.data_dictionary_csv)
    print(artifacts.timestamp_profile_json)
    print(artifacts.failure_timeline_csv)
    print(artifacts.readme)


if __name__ == "__main__":
    main()
