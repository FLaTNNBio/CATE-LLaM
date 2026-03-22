"""
python -m src.BioMistral.calculated_cate `
--provider openrouter `
--model_id mistralai/mistral-7b-instruct-v0.1 `
--dataset_path data/aids/aids_csv.csv `
--prompt_path src/BioMistral/prompts/short_prompt.txt `
--output_path result_cate_llm_BioMistra/\ `
--chunk_size 30 `
--max_retries 0
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate cate")

    #LLM config
    parser.add_argument("--provider", type=str,required=True, default="openrouter", help="LLM provider")
    parser.add_argument("--model_id", type=str,required=True, default="google/gemma-3-27b-it" ,help="model id")


    #Pipeline config
    parser.add_argument("--dataset_path", type=str,required=True, default="data/aids/aids_csv.csv", help="dataset path")
    parser.add_argument("--prompt_path", type=str,required=True, default="src/BioMistral/prompts/bio_prompt.txt", help="prompt path")
    parser.add_argument("--output_path", type=Path,required=True, help="output path")
    parser.add_argument("--chunk_size", type=int,required=False, default=8, help="chunk size")
    parser.add_argument("--max_retries", type=int,required=False, default=3, help="max retries")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    #models
    #      model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    #      model_id = "mistralai/mistral-7b-instruct-v0.1"
    #      model_id="google/gemma-3-27b-it"

    # configuration
    config = AppConfig(
        llm=LLMConfig(
            provider=args.provider,
            model_id=args.model_id,
        ),

        pipeline=PipelineConfig(
            dataset_path=args.dataset_path,
            prompt_path=args.prompt_path,
            output_path=args.output_path,
            chunk_size=args.chunk_size,
            max_retries=args.max_retries
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
        output_dir=config.pipeline.output_path,
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