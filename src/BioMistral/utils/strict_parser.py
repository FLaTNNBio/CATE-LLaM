import json
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StrictJSONParser:
    """
    Robust parser for extracting JSON arrays from LLM output.

    Strategy:
    1. Find all JSON-like arrays in the text
    2. Attempt JSON decoding
    3. Return the first successfully parsed list
    """

    ARRAY_PATTERN = re.compile(r"\[[\s\S]*?\]")

    def parse(self, text: str) -> Optional[List[Dict[str, Any]]]:

        if not text:
            logger.warning("Empty LLM output.")
            return None

        text = text.strip()

        candidates = self.ARRAY_PATTERN.findall(text)

        if not candidates:
            logger.warning("No JSON array detected in LLM output.")
            return None

        for candidate in candidates:

            try:
                parsed = json.loads(candidate)

                if isinstance(parsed, list):
                    return parsed

            except json.JSONDecodeError:
                continue

        logger.warning("No valid JSON array could be parsed.")
        return None