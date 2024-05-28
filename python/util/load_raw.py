"""
File: load_raw.py
Author: Chuncheng Zhang
Date: 2024-04-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load the raw object from the data folder.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
import mne
import pandas as pd

from . import logger


# %% ---- 2024-04-23 ------------------------
# Function and class
class RawObject(object):
    file = None
    raw = None

    def __init__(self, file: pd.Series):
        self.file = file
        self._load_raw()

    def _load_raw(self):
        file = self.file

        if file['format'] == '.bdf':
            raw = mne.io.read_raw(file['path'])
            annotations = mne.read_annotations(file['evt_path'])
            raw.set_annotations(annotations)

        self.raw = raw


# %% ---- 2024-04-23 ------------------------
# Play ground


# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
