"""
File: default.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Make default configuration

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from dataclasses import dataclass
from rich import print


# %% ---- 2024-02-21 ------------------------
# Function and class
root = Path(__file__).parent


@dataclass
class Base:
    generated_date: str = datetime.now()


@dataclass
class Data:
    data_folder: str = Path("d://脑机接口专项").as_posix()


@dataclass
class Conf(Base, Data):
    author: str = "default"


# %% ---- 2024-02-21 ------------------------
# Play ground
if __name__ == "__main__":
    config = OmegaConf.structured(Conf)
    print(f"Configuring: {config}")
    OmegaConf.save(config, root.joinpath("default.yaml"))

# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
