from dataclasses import dataclass


@dataclass(slots=True) #loss memory
class CATEResult:
    original_index: int
    cate_estimate: float
