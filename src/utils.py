import datetime
import glob
import os
from pathlib import Path

import yaml
from pathvalidate import sanitize_filename


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


def get_logging_conf(filename=None, suffix=None):
    with open(get_project_root("logs/config.yaml"), "rt") as f:
        config = yaml.safe_load(f.read())
    if filename is not None:
        filename += "_" + datetime.datetime.now().strftime("%b%d_%H%M_%S%f")
        if suffix is not None:
            filename += "_" + suffix
        filename = sanitize_filename(filename)
        config["handlers"]["file"]["filename"] = f"logs/{filename}.log"
        config["handlers"]["r_file"]["filename"] = f"logs/{filename}.log"
    return config
