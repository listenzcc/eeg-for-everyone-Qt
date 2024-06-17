"""
File: load_raw_data.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Preprocess the data.

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
from .cache_data import CacheData


# %% ---- 2024-02-21 ------------------------
# Function and class
class LoadRawData(CacheData):
    def __init__(self, file_info: dict):
        super().__init__(file_info)
        self.load_raw()
        self.standard_montage()
        self.get_events()
        self.filter_ch_names()

    def filter_ch_names(self):
        self.ch_names_inside_montage = [
            e for e in self.raw.ch_names if e in self.montage.ch_names
        ]
        self.ch_names_outside_montage = [
            e for e in self.raw.ch_names if e not in self.montage.ch_names
        ]
        logger.debug(
            f"Filtered ch_names inside: {self.ch_names_inside_montage} outside: {self.ch_names_outside_montage}"
        )

    def load_raw(self):
        """
        Loads the raw data from a file.

        Returns:
            mne.io.Raw | None: The loaded raw data, or None if loading failed.
        """

        not_loaded = True

        if self.file_info["file_name"] == "data.bdf":
            not_loaded = False
            path = self.file_info["path"]
            raw = mne.io.read_raw(path)
            annotations = mne.read_annotations(path.parent.joinpath("evt.bdf"))
            raw.set_annotations(annotations, verbose=True)
            logger.debug(
                f"Cloned annotations {annotations} from evt.bdf to the raw of data.bdf"
            )
            logger.debug(f"Loaded raw: {raw}")

        if not_loaded:
            logger.warning(f"Failed load from file {self.file_info}")
            return

        self.raw = raw
        return raw

    def get_events(self):
        """
        Extracts events and event IDs from the raw data.

        Returns:
            tuple: A tuple containing the events array and the event ID dictionary.
        """

        events, event_id = mne.events_from_annotations(self.raw)
        self.events = events
        self.event_id = event_id
        logger.debug(f"Got events (shape):{events.shape}, event_id: {event_id}")
        fig = mne.viz.plot_events(
            events,
            sfreq=self.raw.info["sfreq"],
            event_id=event_id,
            show=False,
        )
        fig.suptitle("Raw events")
        fig.savefig(self.to_cache("raw-events.jpg"))
        return events, event_id

    def standard_montage(
        self, montage_name: str = "standard_1020", rename_channels: dict = None
    ):
        """
        Applies a standard montage to the raw data.

        Args:
            montage_name (str, optional): The name of the standard montage to apply. Defaults to "standard_1020".
            rename_channels (dict, optional): A dictionary mapping original channel names to new names. Defaults to None.

        Returns:
            mne.channels.DigMontage: The applied standard montage.
        """

        montage = mne.channels.make_standard_montage(montage_name)

        if rename_channels is not None:
            montage.rename_channels(rename_channels)
            logger.debug(f"Renamed channels at standard montage: {rename_channels}")

        mapping = {n: n.upper() for n in self.raw.ch_names}
        self.raw.rename_channels(mapping)

        mapping = {n: n.upper() for n in montage.ch_names}
        montage.rename_channels(mapping)

        self.montage = montage
        self.raw.set_montage(montage, on_missing="warn")

        logger.debug(f"Applied standard montage: {montage_name} to raw")
        return montage


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending
