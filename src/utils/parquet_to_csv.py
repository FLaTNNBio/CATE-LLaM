import argparse
import os
import pandas as pd


def parquet_to_csv(input_path, output_path, chunksize=None, compression=None):
    print(f"Reading parquet file: {input_path}")

    if chunksize is None:
        # Standard conversion (carica tutto in memoria)
        df = pd.read_parquet(input_path)
        print("Writing CSV...")
        df.to_csv(
            output_path,
            index=False,
            compression=compression
        )
    else:
        # Conversione chunked (memory-safe per file grandi)
        print("Using chunked conversion...")
        parquet_file = pd.read_parquet(input_path)

        total_rows = len(parquet_file)
        for i in range(0, total_rows, chunksize):
            chunk = parquet_file.iloc[i:i+chunksize]
            mode = "w" if i == 0 else "a"
            header = i == 0

            chunk.to_csv(
                output_path,
                mode=mode,
                header=header,
                index=False,
                compression=compression
            )

    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet file to CSV.")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--chunksize", type=int, default=None,
                        help="Optional chunk size for large files (e.g., 100000)")
    parser.add_argument("--compression", choices=["gzip", "bz2", "zip", "xz"],
                        default=None, help="Optional compression for CSV")

    args = parser.parse_args()

    parquet_to_csv(
        input_path=args.input,
        output_path=args.output,
        chunksize=args.chunksize,
        compression=args.compression
    )


if __name__ == "__main__":
    main()