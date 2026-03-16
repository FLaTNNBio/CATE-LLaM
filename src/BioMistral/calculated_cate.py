from pathlib import Path

from src.BioMistral.llm_hf.llm_factory import LLMFactory
from src.BioMistral.utils import (
    StrictJSONParser,
    CATEValidator,
    ResultSaver,
    setup_logging
)
from src.BioMistral.config import AppConfig, LLMConfig, PipelineConfig
from src.BioMistral.domain.cate_estimator import CATEEstimator
from src.BioMistral.pipeline.cate_pipeline import CATEPipeline


def main():
    setup_logging()

    # configuration
    config = AppConfig(
        # llm=LLMConfig(
        #     provider="hf",
        #     model_id="meta-llama/Meta-Llama-3-8B-Instruct"
        # ),

        llm=LLMConfig(
             provider="openrouter",
             model_id="google/gemma-3-27b-it"
        ),

        pipeline=PipelineConfig(
            dataset_path=Path("data/analytic/analytic_sepsis_early_diuretics_v1.parquet"),
            prompt_path=Path("src/BioMistral/prompts/prompt.txt"),
            output_path=Path("result_cate_llm/calculated_cate3.csv"),
            chunk_size=5,
            max_retries=3
        )
    )

    llm = LLMFactory.create(config.llm)
    llm.load()

    # parser + validation
    parser = StrictJSONParser()
    validator = CATEValidator()

    # estimator
    estimator = CATEEstimator(
        llm=llm,
        parser=parser,
        validator=validator,
        config=config.llm
    )

    # result saver
    saver = ResultSaver(
        output_dir=config.pipeline.output_path.parent,
        model_name=config.llm.model_id
    )


    # pipeline
    pipeline = CATEPipeline(
        config.pipeline,
        estimator,
        saver
    )

    pipeline.load_resources()
    pipeline.run()


if __name__ == "__main__":
    main()