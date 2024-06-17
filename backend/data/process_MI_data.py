"""
File: process_MI_data.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Process the MI data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
import mne

from . import logger
from .load_raw_data import LoadRawData


# %% ---- 2024-02-21 ------------------------
# Function and class
class ProcessMIData(LoadRawData):
    def __init__(self, file_info: dict, parameters: dict):
        super().__init__(file_info)
        self.get_epochs()

    def get_epochs(self):
        pass


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
