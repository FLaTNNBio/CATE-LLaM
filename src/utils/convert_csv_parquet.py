# Convert CSV files to Parquet format using pandas and viceversa
from typing import Union

import pandas as pd
from pathlib import Path



def csv_to_parquet(csv_path: Path, parquet_path: Union[Path, None] = None ) -> None:
    """
    Convert a CSV file to Parquet format.
    :param csv_path: path to the input CSV file
    :param parquet_path: path to the output Parquet file
    """
    df = pd.read_csv(csv_path)

    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')

    df.to_parquet(parquet_path, index=False)

def parquet_to_csv(parquet_path: Path, csv_path: Union[Path, None] = None) -> None:
    """
    Convert a Parquet file to CSV format.
    :param parquet_path: path to the input Parquet file
    :param csv_path: path to the output CSV file
    """
    df = pd.read_parquet(parquet_path)

    if csv_path is None:
        csv_path = parquet_path.with_suffix('.csv')

    df.to_csv(csv_path, index=False)

def main():
    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet or vice versa")
    parser.add_argument("--input_path", required=True, type=Path, help="Path to the input file (CSV or Parquet)")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to the output file (optional)")
    parser.add_argument("--to_csv", action="store_true", help="Convert CSV to Parquet (default: convert Parquet to CSV)")
    args = parser.parse_args()

    if args.to_csv:
        if args.input_path.suffix.lower() != ".parquet":
            raise ValueError("Input file must be a Parquet when --to_parquet is not specified")
        parquet_to_csv(args.input_path, args.output_path)
        print(f"Converted {args.input_path} to CSV format at {args.output_path or args.input_path.with_suffix('.csv')}")
    else:
        if args.input_path.suffix.lower() != ".csv":
            raise ValueError("Input file must be a CSV when --to_parquet is specified")
        csv_to_parquet(args.input_path, args.output_path)
        print(f"Converted {args.input_path} to Parquet format at {args.output_path or args.input_path.with_suffix('.parquet')}")

if __name__ == "__main__":
    main()