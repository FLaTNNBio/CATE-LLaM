
from huggingface_hub import InferenceClient

from .base_llm import BaseLLM

class RemoteLLM(BaseLLM):
    """
    Implementazione per modelli remoti via HuggingFace Inference API.
    """

    def __init__(self, model_id: str, api_key: str):
        """
        :param model_id: HF model id (es. meta-llama/Meta-Llama-3-8B-Instruct)
        :param api_key: token HF
        """
        super().__init__(model_id)

        self.api_key = api_key
        self.client = None

    def load(self):
        """
        Inizializza il client HF.
        """

        if not self.api_key:
            raise ValueError("API key is required for RemoteLLM.")

        print(f"Initializing remote model: {self.model_id}")

        self.client = InferenceClient(api_key=self.api_key)

        print("Remote client initialized.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ) -> str:
        """
        Genera testo usando chat_completion.
        """

        if self.client is None:
            raise RuntimeError("Client not initialized. Call load() first.")
        try:

            response = self.client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            raise RuntimeError(f"LLM API error: {e}")
