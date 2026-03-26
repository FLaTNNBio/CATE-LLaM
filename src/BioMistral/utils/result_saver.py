import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Any
from dataclasses import is_dataclass, asdict


class ResultSaver:
    def __init__(self, output_dir: Path, model_name: str):
        """
            Gestore centralizzato per il salvataggio di qualsiasi tipo di dato in CSV.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_model_name = model_name.replace("/", "_")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_prefix = f"{safe_model_name}_{now}"

        # Mappa interna per tracciare i file creati dinamicamente
        self._files: Dict[str, Path] = {}

    def _get_file_path(self, data_type: str) -> Path:
        """Restituisce (e traccia) il percorso del file per un dato tipo di dato."""
        if data_type not in self._files:
            filename = f"{data_type}_{self.file_prefix}.csv"
            self._files[data_type] = self.output_dir / filename
        return self._files[data_type]

    def save(self, data_type: str, records: List[Union[Dict, Any]]):
        """
        Salva una lista di record (dizionari o dataclass) in un file CSV.
        Il file viene creato dinamicamente in base al 'data_type'.
        """
        if not records:
            return

        # Converte le dataclass in dizionari, se necessario
        if is_dataclass(records[0]):
            records = [asdict(r) for r in records]

        df = pd.DataFrame(records)
        file_path = self._get_file_path(data_type)

        # Se il file non esiste ancora, scriveremo anche l'header
        write_header = not file_path.exists()

        df.to_csv(
            file_path,
            mode="a",
            header=write_header,
            index=False
        )

    def load_processed_indices(self, data_type: str = "results", id_column: str = "original_index") -> set:
        """Restituisce l'insieme degli ID già salvati per un certo tipo di dato."""
        file_path = self._get_file_path(data_type)
        if not file_path.exists():
            return set()

        df = pd.read_csv(file_path, usecols=[id_column])
        return set(df[id_column])


    #
    # def __init__(self, output_dir: Path, model_name: str):
    #     """
    #     Gestisce sia il salvataggio dei risultati CATE sia del report CSV.
    #     """
    #     now = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     safe_model_name = model_name.replace("/", "_")
    #
    #     # File principale dei risultati
    #     results_filename = f"calculate_cate_{safe_model_name}_{now}.csv"
    #     self.results_path = output_dir / results_filename
    #     self.results_path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     if not self.results_path.exists():
    #         pd.DataFrame(columns=[
    #             "original_index",
    #             "cate_estimate"
    #         ]).to_csv(self.results_path, index=False)
    #
    #     # File report CSV
    #     report_filename = f"report_{safe_model_name}_{now}.csv"
    #     self.report_path = output_dir / report_filename
    #     if not self.report_path.exists():
    #         pd.DataFrame(columns=[
    #             "id",
    #             "status",
    #             "error",
    #             "retries"
    #         ]).to_csv(self.report_path, index=False)
    #
    # def load_processed_indices(self):
    #     """Restituisce l'insieme degli ID già salvati nel file principale"""
    #     if not self.results_path.exists():
    #         return set()
    #     df = pd.read_csv(self.results_path)
    #     return set(df["original_index"])
    #
    # def save_results(self, results: List[CATEResult]):
    #     """Salva i risultati CATE nel file principale"""
    #     df = pd.DataFrame([asdict(r) for r in results])
    #     df = df[["original_index", "cate_estimate"]]
    #
    #     df.to_csv(
    #         self.results_path,
    #         mode="a",
    #         header=False,
    #         index=False
    #     )
    #
    # def save_report(self, report_rows: List[Dict]):
    #     """
    #     Salva o aggiorna il report CSV.
    #     report_rows: lista di dizionari con campi:
    #         - id
    #         - status ("success", "missing", "retry_failed", "failed")
    #         - error
    #         - retries (numero tentativi)
    #     """
    #     if not report_rows:
    #         return
    #
    #     df = pd.DataFrame(report_rows)
    #     df = df[["id", "status", "error", "retries"]]
    #
    #     df.to_csv(
    #         self.report_path,
    #         mode="a",
    #         header=False,
    #         index=False
    #     )