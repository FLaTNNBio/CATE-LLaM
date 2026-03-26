
from openai import OpenAI
from .base_llm import BaseLLM

class RemoteLLM(BaseLLM):
    """
    Implementanion for remote  model from HuggingFace Inference API and openrouter.
    """

    def __init__(
            self,
            model_id: str,
            api_key: str,
            base_url: str
    ):
        super().__init__(model_id)

        if not api_key:
            raise ValueError("API key is required for remote models")

        if not base_url:
            raise ValueError("base_url is required for remote models")

        self.api_key = api_key
        self.base_url = base_url
        self.client = None


    def load(self):

        self.client = OpenAI(api_key=self.api_key,
                             base_url=self.base_url)
        self._loaded = True
        print("Loaded model: {}".format(self.model_id))
        print("Remote client initialized.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ) -> str:
        self._check_loaded()


        if self.client is None:
            raise RuntimeError("Client not initialized. Call load() first.")
        try:

            response = self.client.chat.completions.create(
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
