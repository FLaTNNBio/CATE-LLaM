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

        print(f"Loading local model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.model.eval()

        print("Local model loaded successfully.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ) -> str:

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return text.strip()