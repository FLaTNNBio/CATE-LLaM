import os
from dotenv import load_dotenv
from .remote_llm import RemoteLLM
from .local_llm import LocalLLM


class LLMFactory:

    @staticmethod
    def create(config):
        load_dotenv()
        if config.provider == "local":

            return LocalLLM(
                model_id=config.model_id,
                device=config.device
            )

        if config.provider == "hf":
            api_key = os.getenv("HF_API_KEY")
            base_url = os.getenv("HF_BASE_URL")

            if not api_key:
                raise ValueError("HF_API_KEY not found in env")
            if not base_url:
                raise ValueError("HF_BASE_URL not found in env")

            return RemoteLLM(
                model_id=config.model_id,
                api_key=api_key,
                base_url=base_url
            )

        if config.provider == "openrouter":

            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in env")
            if not base_url:
                raise ValueError("OPENROUTER_BASE_URL not found in env")

            return RemoteLLM(
                model_id=config.model_id,
                api_key=api_key,
                base_url=base_url
            )

        raise ValueError(f"Unknown provider {config.provider}")