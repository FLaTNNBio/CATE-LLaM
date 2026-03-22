from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for any LLM model.

    All implementations must follow this interface.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False

    @abstractmethod
    def load(self):
        """
        Load the model and initialize it.
        """
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ) -> str:

        pass

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} not loaded. Call load() first."
            )