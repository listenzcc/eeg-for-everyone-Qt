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
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.io import savemat
from dash import dash_table, dcc

from .. import logger, dash_app
from ..load_data.load_epochs import EpochsObject


# %% ---- 2024-06-14 ------------------------
# Function and class
def convert_info_to_table(info):
    df = pd.DataFrame(
        [(k, f'{info[k]}') for k in info],
        columns=['key', 'value']
    )
    df['idx'] = df.index
    df = df[['idx', 'key', 'value']]

    return dash_table.DataTable(
        df.to_dict("records"),
        [{"name": i, "id": i} for i in df.columns],
        filter_action="native",
        filter_options={"placeholder_text": "Filter column..."},
        # Auto wrap lines
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        # left align text in columns for readability
        style_cell={'textAlign': 'left'},
        page_size=20,
    )


class BaseAnalysis(object):
    standard_montage_name = 'standard_1020'
    protocol = 'Any'
    n_jobs = 32
    files = []
    objs = []  # All the chosen files
    options = {}

    # ----------------------------------------
    # ---- Analysis methods ----
    methods = {}

    def __init__(self, protocol: str = None, files: list = None, options: dict = None):
        if files is None:
            files = []

        if options is None:
            options = {}

        if protocol:
            self.protocol = protocol

        self.objs = []
        self.options = {}
        self.files = list(files)

        # Fill options
        self.options |= options

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

        assert objs, 'No raw data available'

        self.objs = objs

    def _load_public_methods(self):
        # TODO Clear the methods
        self.methods.clear()
        self.methods['Plot Events'] = self._method_plot_events
        self.methods['Plot Sensors'] = self._method_plot_sensors
        self.methods['Plot Evoked'] = self._method_plot_evoked

    def _method_plot_events(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs
        dash_app.div.children.append(convert_info_to_table(epochs.info))

        return mne.viz.plot_events(epochs.events, epochs.info['sfreq'], event_id=epochs.event_id, show=False)

    def _method_plot_sensors(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs[selected_event_id]
        dash_app.div.children.append(convert_info_to_table(epochs.info))

        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        mne.viz.plot_sensors(
            epochs.info, show_names=True, axes=axes, show=False)
        return fig

    def _method_plot_evoked(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs[selected_event_id]

        # Plot into the dashboard
        dash_app.div.children.append('Epochs detail')
        for ch_name in self.options['channels']:
            data = epochs.copy().pick([ch_name.upper()]).get_data(copy=False)
            # Squeeze data shape into (trials x times)
            data = data.squeeze()

            # Plot the epochs detail in evoked mean time-series
            # If the data is one-dimensional matrix, plot the time series.
            # If the data is two-dimensional matrix, plot the matrix and plot the mean time series on the first dimension.
            kwargs = dict(title=f'{ch_name}', x=epochs.times)
            if len(data.shape) == 1:
                # Plot time-series
                fig = px.line(y=data, **kwargs)
                dash_app.div.children.append(dcc.Graph(figure=fig))
            else:
                # Plot matrix
                fig = px.imshow(data, aspect='auto', **kwargs)
                dash_app.div.children.append(dcc.Graph(figure=fig))

                # Plot time-series
                fig = px.line(y=np.mean(data, axis=0), **kwargs)
                dash_app.div.children.append(dcc.Graph(figure=fig))

        dash_app.div.children.append(convert_info_to_table(epochs.info))

        evoked: mne.Evoked = epochs.average()
        logger.debug(f'Got evoked: {evoked}')
        return evoked.plot_joint(
            title=f'Evoked: {selected_event_id} | {len(epochs)}',
            show=False,
            exclude=['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'],
        )

    def _save_data(self, selected_idx, selected_event_id, path):
        epochs = self.objs[selected_idx].epochs[selected_event_id]
        data = epochs.get_data()
        times = epochs.times
        ch_names = epochs.ch_names
        package = dict(
            data=data,
            times=times,
            ch_names=ch_names
        )
        try:
            savemat(path, mdict=package, appendmat=True)
            logger.debug(f'Saved epochs {epochs} to path: {path}')
        except Exception as err:
            logger.error(f'Error saving {path}, {err}')
            import traceback
            traceback.print_exc()

# %% ---- 2024-06-14 ------------------------
# Play ground

# %% ---- 2024-06-14 ------------------------
# Pending

# %% ---- 2024-06-14 ------------------------
# Pending
