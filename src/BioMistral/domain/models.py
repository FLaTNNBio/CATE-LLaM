from dataclasses import dataclass


@dataclass(slots=True) #loss memory
class CATEResult:
    original_index: int
    cate_estimate: float
    model_id: str
    temperature: float
    max_tokens: int