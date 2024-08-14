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
import contextlib
import pandas as pd

from .. import logger
from .load_raw import RawObject


# %% ---- 2024-06-14 ------------------------
# Function and class
def convert_df_to_list(df):
    return [e.to_list() for _, e in df.iterrows()]


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
            # Selecting all event_ids
            logger.debug('Selecting all the event_ids as required')

        elif selected_event_id := {
            k: v for k, v in event_id.items() if k in eventIds
        }:
            values = list(selected_event_id.values())
            df = pd.DataFrame(
                events, columns=['timestamp', 'duration', 'event'])
            df['flag'] = df['event'].map(lambda e: e in values)

            # If label.csv exists, trust it
            try:
                src = self.file['path'].parent.joinpath('label.csv')
                label_df = pd.read_csv(src)
                df['label'] = label_df['label']
                event_id = {f'{e}': e for e in df['label'].unique()}
                df = df[df['flag']]
                events = convert_df_to_list(
                    df[['timestamp', 'duration', 'label']])
                logger.warning(
                    f'Default switch: using event_id: {event_id} from customized label.csv: {src}')

            # If using label.csv fails, using selected_event_id instead
            except Exception as e:
                event_id = selected_event_id
                df = df[df['flag']]
                events = convert_df_to_list(
                    df[['timestamp', 'duration', 'event']])
                logger.warning(
                    f'Default switch: Using selected event_id: {event_id}')

        else:
            # Required to select some events, but they are unavailable
            logger.warning(f'Can not find any event_ids in {eventIds}')

        if not events:
            logger.warning('No events found')

        logger.debug(
            f'Found event_id: {event_id}, events: {len(events)} records')

        # ----------------------------------------
        # ---- Make kwargs ----
        # Picks
        if channels := options.get('channels'):
            kwargs = dict(picks=[e.upper() for e in channels])
        else:
            kwargs = dict(picks=['eeg'])
        # epochTimes: tmin, tmax, ...
        kwargs |= options.get('epochTimes', {})
        # epochsKwargs: baseline, detrend, decim, event_repeated, ...
        kwargs |= options.get('epochsKwargs', {})

        # Discard baseline if tmin is larger than 0
        if kwargs.get('tmin') > 0:
            baseline = kwargs['baseline']
            kwargs['baseline'] = (None, None)
            logger.warning(
                f'Forcedly reset baseline from {baseline} to (None, None) since tmin is greater than 0')

        if reject := options.get('reject'):
            kwargs |= dict(reject=reject)

        logger.debug(f'Getting epochs with kwargs: {kwargs}')

        # ----------------------------------------
        # ---- Reset reference ----
        self.raw.load_data()
        if ref_channels := options.get('otherOptions', {}).get('ref_channels'):
            ref_channels = [e.upper() for e in ref_channels]
            self.raw.set_eeg_reference(ref_channels=ref_channels)
            logger.debug(f'Changed reference channels to {ref_channels}')
        else:
            logger.warning(
                'Ignore reset channels, since not set ref_channels option.')

        # ----------------------------------------
        # ---- Filter ----
        filter_kwargs = options.get('freqBand')
        filter_kwargs |= dict(picks=kwargs['picks'])
        filter_kwargs |= dict(n_jobs=16)
        self.raw.filter(**filter_kwargs)

        # ----------------------------------------
        # ---- Fetch epochs ----
        epochs = mne.Epochs(self.raw, events, event_id, preload=True, **kwargs)

        logger.debug(f'Got epochs: {epochs}')
        self.epochs = epochs


# %% ---- 2024-06-14 ------------------------
# Play ground


# %% ---- 2024-06-14 ------------------------
# Pending


# %% ---- 2024-06-14 ------------------------
# Pending
