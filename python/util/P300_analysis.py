"""
File: P300_analysis.py
Author: Chuncheng Zhang
Date: 2024-06-17
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis for P300 Dataset

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-17 ------------------------
# Requirements and constants
import mne
import numpy as np
import matplotlib.pyplot as plt

from . import logger
from .analysis.base_analysis import BaseAnalysis
from .default.n_jobs import n_jobs

# %% ---- 2024-06-17 ------------------------
# Function and class


class P300_Analysis(BaseAnalysis):
    protocol = 'P300'

    def __init__(self, protocol: str = None, files: list = None, options: dict = None):
        if files is None:
            files = []
        if options is None:
            options = {}
        super(P300_Analysis, self).__init__(protocol, files, options)
        self.load_methods()

    def load_methods(self):
        self.methods['debug'] = self.debug

    def debug(self, idx, event_id):
        epochs = self.objs[idx].epochs[event_id]
        sfreq = epochs.info['sfreq']

        # Data shape is (trials, channels, time-points)
        data = epochs.get_data()
        print(data.shape, sfreq)


# %% ---- 2024-06-17 ------------------------
# Play ground


# %% ---- 2024-06-17 ------------------------
# Pending


# %% ---- 2024-06-17 ------------------------
# Pending
