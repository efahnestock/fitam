

from dataclasses import dataclass
import logging


@dataclass
class LoggingConfig:

    file_logging_level: int = logging.DEBUG
    console_logging_level: int = logging.WARNING
    logging_filename: str = 'main_output.log'
    logging_format: str = '%(asctime)s |%(levelname)s %(relativeCreated)6d ProcessID: %(process)d | %(message)s'
