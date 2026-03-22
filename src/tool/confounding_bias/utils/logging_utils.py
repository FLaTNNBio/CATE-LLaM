from __future__ import annotations

import logging


def setup_logging(verbose: bool) -> None:
    """
    Configure application logging.

    Parameters
    ----------
    verbose : bool
        If True, use INFO level. Otherwise use WARNING level.
    """
    level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )