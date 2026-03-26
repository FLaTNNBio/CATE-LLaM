import logging


logger = logging.getLogger(__name__)


class CATEEstimator:

    def __init__(self, llm, parser, validator, config):
        self.llm = llm
        self.parser = parser
        self.validator = validator
        self.config = config

    def estimate(self, system_prompt, user_prompt, expected_ids):

        remaining_ids = set(expected_ids)
        final_results  = {}

        for retries in range(self.config.max_retries):
            if not remaining_ids:
                break #all id

            prompt = user_prompt
            if retries > 0:
                prompt = self._reinforce_prompt(user_prompt, retries, remaining_ids)


            raw_output = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            logger.info(f"OUTPUT:\n{raw_output}")
            logger.info(f"LLM RAW OUTPUT:\n{raw_output}")

            try:
                parsed = self.parser.parse(raw_output)

                valid, reason, partial_report = self.validator.validate(parsed,remaining_ids)

                # aggiorna risultati e logga missing
                final_results.update(partial_report["valid_items"])
                remaining_ids = set(remaining_ids) - set(final_results.keys())

                logger.warning(f"Missing IDs after attempt {retries}: {remaining_ids}")

                if not valid:
                    logger.warning(f"Validation failed: {reason}")
                    raise ValueError(f"Validation failed: {reason}")

                if not remaining_ids:
                    break

            except Exception as e:
                logger.warning(f"Attempt {retries} failed: {e}")

        if remaining_ids:
            logger.warning(f"Some IDs could not be estimated: {remaining_ids}")

        return final_results


    def _reinforce_prompt(self, prompt, retries, missing_ids):

        missing_str = ", ".join(str(i) for i in missing_ids)
        return (
                prompt +
                f"\n\nWARNING: Previous output was INVALID.\n"
                f"Retry attempt {retries}.\n"
                f"Please provide estimates for these IDs only: {missing_str}\n"
                "You MUST follow JSON format EXACTLY.\n"
                "Any deviation will be rejected.\n"
        )
