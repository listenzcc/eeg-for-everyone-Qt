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

            # with contextlib.suppress(Exception):
            # If label.csv exists, trust it
            # else, using selected_event_id
            try:
                src = self.file['path'].parent.joinpath('label.csv')
                label_df = pd.read_csv(src)
                df['label'] = label_df['label']
                event_id = {f'{e}': e for e in df['label'].unique()}
                df = df[df['flag']]
                events = convert_df_to_list(
                    df[['timestamp', 'duration', 'label']])
                logger.warning(f'Using event_id: {event_id} from {src}')

            except Exception as e:
                import traceback
                traceback.print_exc()

                event_id = selected_event_id
                df = df[df['flag']]
                events = convert_df_to_list(
                    df[['timestamp', 'duration', 'event']])

            # events, event_id = mne.events_from_annotations(
            #     self.raw, selected_event_id)
        else:
            # Required to select some events, but they are unavailable
            logger.warning(f'Can not find any event_ids in {eventIds}')

        # else:
        #     df = pd.DataFrame(events, columns=['timestamp', 'duration', 'event'])
        #     label_df = None
        #     try:
        #         label_df = pd.read_csv(self.file['path'].parent.joinpath('label.csv'))
        #     except Exception as e:
        #         pass
        #     print(label_df)

        #     if label_df:
        #         df['label'] = label_df['label']

        #     print(df)

        # elif selected_event_id := {
        #     k: v for k, v in event_id.items() if k in eventIds
        # }:
        #     events, event_id = mne.events_from_annotations(
        #         self.raw, selected_event_id)
        # else:
        #     logger.warning('Can not find any event_ids being selected')

        logger.debug(
            f'Found event_id: {event_id}, events: {len(events)} records')

        # ----------------------------------------
        # ---- Make kwargs ----
        kwargs = dict(picks=['eeg'], detrend=1, event_repeated='drop')
        kwargs |= options.get('epochTimes', {})
        kwargs |= options.get('epochsKwargs', {})

        if reject := options.get('reject'):
            kwargs |= dict(reject=reject)

        logger.debug(f'Getting epochs with kwargs: {kwargs}')

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
