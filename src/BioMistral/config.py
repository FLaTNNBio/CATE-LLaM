from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:

    provider: str = "hf"  # hf | openrouter | local

    model_id: str = None

    temperature: float = 0.5
    max_tokens: int = 800

    device: str = "cuda"

    api_key: str | None = None
    base_url: str | None = None


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