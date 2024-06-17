"""
File: base_analysis.py
Author: Chuncheng Zhang
Date: 2024-06-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

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
import numpy as np
import matplotlib.pyplot as plt

from . import logger
from .load_epochs import EpochsObject


# %% ---- 2024-06-14 ------------------------
# Function and class
class BaseAnalysis(object):
    standard_montage_name = 'standard_1020'
    protocol = 'Any'
    n_jobs = 32
    files = []
    objs = []
    options = {}

    # ----------------------------------------
    # ---- Analysis methods ----
    methods = {}

    def __init__(self, protocol: str = None, files: list = [], options: dict = {}):
        if protocol:
            self.protocol = protocol

        # Clear the self.files is necessary, since the class is reuseable
        self.files = []
        self.objs = []
        self.options = {}

        # Fill files
        self.files.extend(files)

        # Fill options
        self.options.update(options)

        # Working pipeline, fill objs
        self.pipeline()

        logger.debug(
            f'Initializing with {self.protocol}, {self.files}, {self.options}')

    def pipeline(self):
        self.load_raws()
        self._load_public_methods()

    def load_raws(self):
        objs = [EpochsObject(file)
                for file in self.files]

        for obj in objs:
            obj._reload_montage(
                self.standard_montage_name,
                self.options.get('rename_channels'))

            obj.get_epochs(self.options)

        assert len(objs) > 0, 'No raw data available'

        self.objs = objs

    def _load_public_methods(self):
        self.methods['Plot Events'] = self._method_plot_events
        self.methods['Plot Sensors'] = self._method_plot_sensors
        self.methods['Plot Evoked'] = self._method_plot_evoked

    def _method_plot_events(self, idx, event_id):
        epochs = self.objs[idx].epochs
        fig = mne.viz.plot_events(
            epochs.events, epochs.info['sfreq'], show=False)
        return fig

    def _method_plot_sensors(self, idx, event_id):
        epochs = self.objs[idx].epochs[event_id]
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        mne.viz.plot_sensors(
            epochs.info, show_names=True, axes=axes, show=False)
        return fig

    def _method_plot_evoked(self, idx, event_id):
        epochs = self.objs[idx].epochs[event_id]
        evoked: mne.Evoked = epochs.average()
        logger.debug(f'Got evoked: {evoked}')
        fig = evoked.plot_joint(
            title=f'Evoked: {event_id} | {len(epochs)}',
            show=False,
            exclude=['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'])
        return fig


# %% ---- 2024-06-14 ------------------------
# Play ground


# %% ---- 2024-06-14 ------------------------
# Pending


# %% ---- 2024-06-14 ------------------------
# Pending
