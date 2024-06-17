"""
File: search_data.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Search available files for EEG data

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
import os

from pathlib import Path
from rich import print, inspect

from .toolbox import md5_encode
from . import logger


# %% ---- 2024-02-21 ------------------------
# Function and class
def check_rules():
    """
    Returns a dictionary of allowed file names and their corresponding check functions.

    Returns:
        dict: A dictionary where the keys are allowed file names and the values are the corresponding check functions.
    """

    allowed_names = {}

    # -------------------------------------------------------
    def _check_data_bdf(path: Path) -> bool:
        b1 = path.is_file()
        b2 = path.parent.joinpath("evt.bdf").is_file()
        return all((b1, b2))

    name = "data.bdf"
    allowed_names[name] = _check_data_bdf

    logger.debug(f"Using check rules: {allowed_names}")
    return allowed_names


class RawDataFiles(object):
    check_rules = check_rules()

    def __init__(self, root: Path):
        root = Path(root)
        assert root.is_dir(), f"Invalid directory: {root}"
        self.root = root

    def _data_info(self, path: Path):
        """
        Returns a dictionary containing information about the given file path.

        Args:
            path (Path): The path to the file.

        Returns:
            dict: A dictionary containing the following information:
                - experiment: The name of the experiment.
                - subject: The name of the subject.
                - file_name: The name of the file.
                - path (Path): The full path to the file.
                - unique: The MD5 hash of the file path.
        """

        parts = path.relative_to(self.root).parts
        assert len(parts) > 2, f"Invalid path: {path}, it at least has Three parts"
        return dict(
            experiment=parts[0],
            subject=parts[1],
            file_name=parts[-1],
            path=path,
            unique=f"{'-'.join(parts[:-1])}-{md5_encode(path.as_posix())}",
        )

    def find_all(self):
        """
        Finds all the files that match the specified rules within the root directory.

        Returns:
            list: A list of file paths that match the rules.
        """

        found = list(os.walk(self.root))
        logger.debug(f"Found {len(found)} folders with files.")
        files = []
        for folder, _, names in found:
            for name in names:
                if name in self.check_rules:
                    path = Path(folder, name)
                    if self.check_rules[name](path):
                        files.append(self._data_info(path))
        logger.debug(f"Found {len(files)} eeg files.")
        return files


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
