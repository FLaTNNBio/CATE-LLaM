import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from dataclasses import asdict

from src.BioMistral.domain.models import CATEResult


class ResultSaver:
    def __init__(self, output_dir: Path, model_name: str):

        now = datetime.now().strftime("%H_M_%S")

        safe_model_name = model_name.replace("/", "_")

        filename = f"calculate_cate_{safe_model_name}_{now}.csv"

        self.output_path = output_dir / filename
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.output_path.exists():
            pd.DataFrame(columns=[
                "original_index",
                "cate_estimate"
            ]).to_csv(self.output_path, index=False)

    def load_processed_indices(self):
        if not self.output_path.exists():
            return set()

        df = pd.read_csv(self.output_path)
        return set(df["original_index"])

    def save(self, results: List[CATEResult]):
        df = pd.DataFrame([asdict(r) for r in results])

        df = df[["original_index", "cate_estimate"]]

        df.to_csv(
            self.output_path,
            mode="a",
            header=False,
            index=False
        )