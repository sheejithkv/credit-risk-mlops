from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def configure_logging(path: str | Path = "configs/logging.yaml") -> None:
    config_path = Path(path)

    if not config_path.exists():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return

    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    logging.config.dictConfig(config)
