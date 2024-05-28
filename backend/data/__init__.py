"""
File: __init__.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Initialize the data module

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

from loguru import logger


# %% ---- 2024-02-21 ------------------------
# Function and class

logger.add(Path(f'log/{datetime.now().strftime("%Y-%m-%d")}.log'))
logger.debug("Initializing data module")


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
