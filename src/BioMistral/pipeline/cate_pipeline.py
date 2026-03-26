import pandas as pd
import logging
import time
import os

from src.BioMistral.domain import CATEResult
from src.BioMistral.utils.chunk_iterator import ChunkIterator

logger = logging.getLogger(__name__)


class CATEPipeline:

    def __init__(self, config, estimator, saver):

        self.config = config
        self.estimator = estimator
        self.saver = saver

        self.df = None
        self.base_prompt = None

    def load_resources(self):

        logger.info("Loading dataset and prompt")
        ext = os.path.splitext(self.config.dataset_path)[1].lower()

        if ext == ".parquet":
            self.df = pd.read_parquet(self.config.dataset_path)
        elif ext == ".csv":
            self.df = pd.read_csv(self.config.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        with open(self.config.prompt_path, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

    def build_prompt(self, chunk):

        #Reset index and assign deterministic IDs
        chunk = chunk.copy()
        if "id" not in chunk.columns:
            raise RuntimeError("Missing 'id' column in chunk")

        expected_ids = set(chunk["id"].tolist())
        csv_input = chunk.to_csv(index=False)
        logger.error(f"INPUT:\n{csv_input}")
        #Load base prompt template (already read into self.base_prompt)
        # Replace placeholders with actual data
        prompt = self.base_prompt \
            .replace("{row_data}", csv_input) \
            .replace("{expected_ids}", str(sorted(expected_ids)))

        return prompt, expected_ids

    def run(self):

        self.load_resources()

        self.df = self.df.reset_index(drop=True)
        self.df["id"] = self.df.index

        all_input_ids = set(self.df["id"])

        processed_list = self.saver.load_processed_indices(data_type="results", id_column="original_index")
        processed_input_ids = set(processed_list)

        iterator = ChunkIterator(
            self.df,
            self.config.chunk_size,
            processed_list
        )

        for chunk in iterator:
            prompt, expected_ids = self.build_prompt(chunk)
            retries = 0
            success = False

            while retries < self.config.max_retries and not success:
                estimates = self.estimator.estimate(
                    "Deterministic clinical engine.",
                    prompt,
                    expected_ids=expected_ids
                )

                report_rows_chunk = []

                if estimates and len(estimates) > 0:
                    results = [
                        CATEResult(
                            original_index=idx,
                            cate_estimate=val,
                        )
                        for idx, val in estimates.items()
                    ]

                    self.saver.save("result", results)
                    processed_input_ids.update(estimates.keys())
                    success = True
                    logger.info(f"Processed IDs: {list(estimates.keys())}")

                    for idx in expected_ids:
                        status = "success" if idx in estimates else "missing"
                        error_msg = "" if idx in estimates else "LLM did not provide value"
                        report_rows_chunk.append({
                            "id": idx, "status": status, "error": error_msg, "retries": retries
                        })

                else:
                    # Nessun output generato, aumenta retry
                    retries += 1
                    logger.warning(f"Retry {retries}/{self.config.max_retries} for chunk {list(chunk.index)}")
                    time.sleep(2)
                    for idx in expected_ids:
                        if idx not in processed_input_ids:
                            report_rows_chunk.append({
                                "id": idx,
                                "status": "retry_failed",
                                "error": f"Attempt {retries} failed",
                                "retries": retries
                            })
                self.saver.save("report", report_rows_chunk)

                if not success:
                    logger.error(f"Chunk {list(chunk.index)} failed permanently, skipping...")
                    # Aggiorna report per tutti gli ID rimasti del chunk
                    fail_rows = [
                        {
                            "id": idx,
                            "status": "failed",
                            "error": "All retries failed",
                            "retries": retries
                        }
                        for idx in expected_ids if idx not in processed_input_ids
                    ]
                    self.saver.save("report", fail_rows)
                    continue

            if not success:
                # invece di fermarsi, logga l'errore e continua
                logger.error(f"Chunk {list(chunk.index)} failed permanently, skipping...")
                continue  # passa al prossimo


        if processed_input_ids != all_input_ids:
            missing = all_input_ids - processed_input_ids
            extra = processed_input_ids - all_input_ids

            raise RuntimeError(
                f"INTEGRITY FAILURE\nMissing: {missing}\nExtra: {extra}"
            )

        logger.info("All patients processed_list successfully. Integrity check passed.")