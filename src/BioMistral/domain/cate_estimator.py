import logging
from typing import List, Optional

from .models import CATEResult

logger = logging.getLogger(__name__)


class CATEEstimator:

    def __init__(self, llm, parser, validator, config):

        self.llm = llm
        self.parser = parser
        self.validator = validator
        self.config = config

    def estimate(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_indices: List[int]
    ) -> Optional[List[CATEResult]]:

        raw_output = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        parsed = self.parser.parse(raw_output)

        # parsing failed
        if parsed is None:
            logger.warning("Parser returned None")
            return None

        valid, reason = self.validator.validate(parsed, expected_indices)

        if not valid:
            logger.warning(f"Validation failed: {reason}")
            return None

        results = []

        for item in parsed:

            results.append(
                CATEResult(
                    original_index=item["patient_index"],
                    cate_estimate=float(item["cate"]),
                    model_id=self.llm.model_id,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            )

        return results