"""
File: cache_data.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Cache system at runtime operations.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
from omegaconf import OmegaConf
from pathlib import Path

from . import logger


# %% ---- 2024-02-21 ------------------------
# Function and class
class CacheData(object):
    def __init__(self, file_info: dict):
        self.file_info = file_info.copy()
        self.init_cache()

    def to_cache(self, subpath: Path) -> Path:
        """
        Returns the path to a file within the cache directory.

        Args:
            subpath (Path): The subpath of the file within the cache directory.

        Returns:
            Path: The path to the file within the cache directory.
        """

        subpath = Path(subpath)
        res = self.file_info["cache_path"].joinpath(subpath)
        res.parent.mkdir(exist_ok=True, parents=True)
        return res

    def init_cache(self):
        """
        Initializes the cache directory for the file.

        Returns:
            Path: The path to the cache directory.
        """

        cache = Path("cache", self.file_info["unique"])
        cache.mkdir(exist_ok=True, parents=True)
        self.file_info["cache_path"] = cache

        conf = OmegaConf.create(self.file_info)
        OmegaConf.save(conf, self.to_cache("file_info.yaml"))

        logger.debug(f"Using cache: {self.file_info}")
        return cache


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
