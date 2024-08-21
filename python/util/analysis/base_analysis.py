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

from nilearn import plotting
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.io import savemat
from dash import dash_table, dcc

from .. import logger, dash_app, asset_path
from ..load_data.load_epochs import EpochsObject


# %% ---- 2024-06-14 ------------------------
# Function and class

# ----------------------------------------
# ---- Read MNI positions ----


def read_known_channel_positions() -> dict:
    p = asset_path.joinpath('MNI-system/eeg-1010-positions.csv')
    df_mni_positions = pd.read_csv(p, header=0, index_col=None)
    print(df_mni_positions)
    known_channel_positions = {}
    for i, se in df_mni_positions.iterrows():
        xyz = (se['X'], se['Y'], se['Z'])
        known_channel_positions[se['Name'].upper()] = xyz
    logger.debug(
        f'Read known channel positions: {known_channel_positions}, file path is {p}')
    return known_channel_positions


known_channel_positions: dict = read_known_channel_positions()


# %%
def convert_info_to_table(info: dict) -> dash_table.DataTable:
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
        self.methods['Plot Connectivity'] = self._method_plot_connectivity

    def _method_plot_connectivity(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs
        evoked = epochs[selected_event_id].average()

        # data shape is (channels x time points)
        data = evoked.data

        ch_names = epochs.info['ch_names']

        # Get channel positions from known_channel_positions
        # ! If the channel name is not available, using (0, 0, 0) instead.
        ch_positions = [(k, known_channel_positions.get(k, (0, 0, 0)))
                        for k in ch_names]
        logger.debug(f'Using ch_positions: {ch_positions}')

        # Raise warning if channel positions are not available
        if not_found_position_channels := [
            k for k in ch_names if k not in known_channel_positions
        ]:
            logger.warning(
                f'Not found position channels: {not_found_position_channels}')

        # Plot matrix
        title = f'Connectivity {selected_event_id}'
        mat = np.corrcoef(data)
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        plotting.plot_matrix(
            mat=mat, labels=ch_names, vmin=-1, vmax=1, title=title, axes=axes)

        # Plot dash app
        node_coords = [e[1] for e in ch_positions]
        view = plotting.view_connectome(
            adjacency_matrix=mat, node_coords=node_coords)

        # Put the embed into dash_app
        from dash import html
        dash_app.dynamic_html = view.get_iframe(width=800, height=600)
        dash_app.div.children.append(html.Embed(
            src='/get_dynamic_html', width=800, height=800))

        # dash_app.div.children.append(dcc.Graph(figure=view))

        return fig

    def _method_plot_events(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs

        dash_app.div.children.append(convert_info_to_table(epochs.info))

        return mne.viz.plot_events(epochs.events, epochs.info['sfreq'], event_id=epochs.event_id, show=False)

    def _method_plot_sensors(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs[selected_event_id]

        # ----------------------------------------
        # ---- Plot channels in the glass brain ----

        ch_names = epochs.info['ch_names']

        # Get channel positions from known_channel_positions
        # ! If the channel name is not available, using (0, 0, 0) instead.
        ch_positions = [(k, known_channel_positions.get(k, (0, 0, 0)))
                        for k in ch_names]
        logger.debug(f'Using ch_positions: {ch_positions}')

        # Raise warning if channel positions are not available
        if not_found_position_channels := [
            k for k in ch_names if k not in known_channel_positions
        ]:
            logger.warning(
                f'Not found position channels: {not_found_position_channels}')

        # Plot dash app
        node_coords = [e[1] for e in ch_positions]
        view = plotting.view_markers(
            marker_coords=node_coords, marker_labels=ch_names)

        # Put the embed into dash_app
        from dash import html
        dash_app.dynamic_html = view.get_iframe(width=800, height=600)
        dash_app.div.children.append(html.Embed(
            src='/get_dynamic_html', width=800, height=800))

        # Make Qt fig
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
