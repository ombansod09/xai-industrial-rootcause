import argparse

from xai_industrial.data.load import convert_to_parquet


p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--metadata-output")
p.add_argument("--source-identifier")
a = p.parse_args()
convert_to_parquet(
    a.input,
    a.output,
    metadata_path=a.metadata_output,
    source_identifier=a.source_identifier,
)
print(a.output)
if a.metadata_output:
    print(a.metadata_output)
