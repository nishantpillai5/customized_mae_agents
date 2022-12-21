import glob
import os
from pathlib import Path


def get_project_root(plus=None):
    """Returns project directory

    Parameters
    ----------
    plus : str, optional
        Join additional path, by default None

    Returns
    -------
    Path
        Path to project root joined with `plus` path
    """
    root = Path(__file__).parent.parent
    if plus is None:
        return root
    else:
        return os.path.join(root, plus)


def get_files(glob_path):
    """Returns list of filepaths in glob string pattern

    Parameters
    ----------
    glob_path : str
        Glob string

    Returns
    -------
    list
        List of filepaths in glob string pattern
    """
    return [os.path.normpath(file) for file in glob.glob(glob_path)]


def get_logfile_path(log_filename):
    """Get filepath to logger file

    Parameters
    ----------
    log_filename : str
        Log file name

    Returns
    -------
    str
        Filepath to log file
    """
    return os.path.join(get_project_root(), "logs", f"{log_filename}.log")


def get_logging_conf():
    """Returns logging configuration

    Returns
    -------
    dict
        logging configuration
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "pipeline": {
                "handlers": ["default", "pipeline_file"],
                "level": "INFO",
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": ["default"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
            "pipeline_file": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": get_logfile_path("pipeline"),
                "maxBytes": 10485760,
                "backupCount": 10,
            },
        },
    }
