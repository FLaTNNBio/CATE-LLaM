from typing import List, Dict, Tuple


class CATEValidator:
    """
    Structural and semantic validation for CATE outputs.
    """

    def validate(
        self,
        parsed: List[Dict],
        expected_indices: List[int]
    ) -> Tuple[bool, str]:

        if parsed is None:
            return False, "Parsing failed"

        if not isinstance(parsed, list):
            return False, "Output is not a list"

        if len(parsed) != len(expected_indices):
            return False, "Length mismatch"

        seen_indices = set()

        for item in parsed:

            if not isinstance(item, dict):
                return False, "Element is not a dict"

            if "patient_index" not in item:
                return False, "Missing patient_index"

            if "cate" not in item:
                return False, "Missing cate"

            idx = item["patient_index"]
            cate = item["cate"]

            if not isinstance(idx, int):
                return False, "patient_index must be int"

            if idx not in expected_indices:
                return False, "Unexpected patient_index"

            if idx in seen_indices:
                return False, "Duplicate patient_index"

            seen_indices.add(idx)

            if not isinstance(cate, (int, float)):
                return False, "cate must be numeric"

            if not (-1.0 <= float(cate) <= 1.0):
                return False, "cate out of bounds"

        return True, "OK"