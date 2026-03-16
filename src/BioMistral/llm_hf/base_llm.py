from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstrac class for anything model


    All implementation do that interfact
    Tutte le implementazioni devono rispettare questa interfaccia.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False

    @abstractmethod
    def load(self):
        """
        Carica il modello o inizializza il client remoto.
        Deve essere chiamato prima di generate().
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