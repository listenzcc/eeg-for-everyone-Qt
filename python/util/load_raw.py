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
    montage = None

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

    def _reload_montage(self, standard_montage_name: str = None, rename_channels: dict = None):
        # If standard_montage_name is not specified, doing nothing
        if standard_montage_name is None:
            return self.raw

        montage = mne.channels.make_standard_montage(standard_montage_name)
        logger.debug(f'Using standard montage: {standard_montage_name}')

        # Rename standard montage's channel names
        if rename_channels is not None:
            montage.rename_channels(rename_channels)
            logger.debug(
                f'Renamed standard montage channel names: {rename_channels}')

        # Rename the channel names as their upper
        self.raw.rename_channels({n: n.upper() for n in self.raw.ch_names})
        montage.rename_channels({n: n.upper() for n in montage.ch_names})

        # Set the montage to the raw
        self.raw.set_montage(montage, on_missing='warn')
        self.montage = montage

        return self.raw


# %% ---- 2024-04-23 ------------------------
# Play ground


# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
