from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Classe astratta base per qualsiasi modello LLM
    (locale o remoto).

    Definisce l'interfaccia comune che deve essere
    rispettata da tutte le implementazioni concrete.
    """

    def __init__(self, model_id: str):
        """
        :param model_id: nome del modello (HF repo id o path locale)
        """
        self.model_id = model_id

    @abstractmethod
    def load(self):
        """
        Carica il modello (inizializzazione client o pesi).
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
        """
        Genera una risposta testuale dal modello.

        :param system_prompt: ruolo system
        :param user_prompt: prompt utente
        :param temperature: temperatura di generazione
        :param max_tokens: massimo numero di token generati
        :return: stringa prodotta dal modello
        """
        pass


