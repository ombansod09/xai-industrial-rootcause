import argparse
from xai_industrial.data.load import convert_to_parquet
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
convert_to_parquet(a.input, a.output)
print(a.output)
