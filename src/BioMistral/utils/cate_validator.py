from typing import List, Dict, Tuple

from src.BioMistral.domain.cate_estimator import logger


class CATEValidator:
    """
    Structural and semantic validation for CATE outputs.
    """

    def validate(
        self,
        parsed: Dict[int, float],
        expected_ids: set
        ) -> Tuple[bool, str, Dict]:

        if not isinstance(parsed, dict):
            return False, "Parsed output must be a dictionary",{}

        valid_items = {}
        errors = []
        for _id, value in parsed.items():

            if not isinstance(value, float):
                errors.append(f"Invalid type for ID {_id}")
                continue

            if value != value:
                errors.append(f"NaN at ID {_id}")
                continue

            if value in (float("inf"), float("-inf")):
                errors.append(f"Infinite at ID {_id}")
                continue

            if not (-1.0 <= value <= 1.0):
                errors.append(f"Out of range ID {_id}")
                continue

            valid_items[_id] = value

        missing_ids = expected_ids - set(valid_items.keys())

        return True, "OK", {
            "valid_items": valid_items,
            "missing_ids": missing_ids,
            "errors": errors
        }
