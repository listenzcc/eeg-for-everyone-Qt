"""
File: load_epochs.py
Author: Chuncheng Zhang
Date: 2024-06-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load epochs from RawObject.
    The options are required to load the epochs

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-14 ------------------------
# Requirements and constants
import mne
import pandas as pd

from .. import logger
from .load_raw import RawObject


# %% ---- 2024-06-14 ------------------------
# Function and class
class EpochsObject(RawObject):
    epochs = None

    def __init__(self, file: pd.Series):
        super().__init__(file)

    def get_epochs(self, options: dict):
        events, event_id = mne.events_from_annotations(self.raw)

        # Find all the selected event_ids if it is provided.
        # But, if 'all' is in the list, using event_ids.
        # still, if can not find any event_ids of selected, using all event_ids instead.
        eventIds = options.get('eventIds', ['all'])

        # ----------------------------------------
        # ---- Select eventIds ----
        if 'all' in eventIds:
            logger.debug('Selecting all the event_ids as required')

        elif selected_event_id := {
            k: v for k, v in event_id.items() if k in eventIds
        }:
            events, event_id = mne.events_from_annotations(
                self.raw, selected_event_id)
        else:
            logger.warning('Can not find any event_ids being selected')

        logger.debug(
            f'Found event_id: {event_id}, events: {len(events)} records')

        kwargs = dict(picks=['eeg'], detrend=0, event_repeated='drop')
        kwargs |= options.get('epochTimes', {})
        kwargs |= options.get('epochsKwargs', {})

        if reject := options.get('reject'):
            kwargs |= dict(reject=reject)

        logger.debug(f'Getting epochs with kwargs: {kwargs}')

        epochs = mne.Epochs(self.raw, events, event_id, preload=True, **kwargs)

        logger.debug(f'Got epochs: {epochs}')
        self.epochs = epochs


# %% ---- 2024-06-14 ------------------------
# Play ground


# %% ---- 2024-06-14 ------------------------
# Pending


# %% ---- 2024-06-14 ------------------------
# Pending
