from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    model_id: str
    temperature: float = 0.1
    max_tokens: int = 800
    device : str = "cuda"


@dataclass
class PipelineConfig:
    dataset_path: Path
    prompt_path: Path
    output_path: Path
    chunk_size: int = 8
    max_retries: int = 3


@dataclass
class AppConfig:
    llm: LLMConfig
    pipeline: PipelineConfig