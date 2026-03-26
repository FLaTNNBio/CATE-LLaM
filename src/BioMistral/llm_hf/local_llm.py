import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_llm import BaseLLM


class LocalLLM(BaseLLM):

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)

        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self._loaded = True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ) -> str:

        self._check_loaded()

        prompt = f"{system_prompt}\n\n{user_prompt}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)