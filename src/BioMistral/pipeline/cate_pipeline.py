import pandas as pd
import logging
import time

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

        self.df = pd.read_parquet(self.config.dataset_path)

        with open(self.config.prompt_path, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

    def build_prompt(self, chunk):

        chunk_text = ""

        for idx, row in chunk.iterrows():
            chunk_text += f"\nPatient index: {idx}\n"
            chunk_text += row.to_string()
            chunk_text += "\n---\n"

        full_prompt = self.base_prompt.replace("{row_data}", chunk_text)
        full_prompt += f"\nYou must output exactly {len(chunk)} results."

        return full_prompt

    def run(self):

        processed = self.saver.load_processed_indices()

        iterator = ChunkIterator(
            self.df,
            self.config.chunk_size,
            processed
        )

        for chunk in iterator:

            expected_indices = list(chunk.index)
            prompt = self.build_prompt(chunk)

            attempts = 0
            success = False

            while attempts < self.config.max_retries and not success:

                results = self.estimator.estimate(
                    "Deterministic clinical engine.",
                    prompt,
                    expected_indices
                )

                if results:
                    self.saver.save(results)
                    success = True
                    logger.info(f"Saved {len(results)} patients")
                else:
                    attempts += 1
                    logger.warning(f"Retry {attempts}")
                    time.sleep(2)

            if not success:
                logger.error(f"Chunk {expected_indices} failed permanently")