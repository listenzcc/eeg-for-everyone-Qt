"""
File: MI_analysis.py
Author: Chuncheng Zhang
Date: 2024-06-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis for Motion Imaging Dataset

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-12 ------------------------
# Requirements and constants
import mne
import numpy as np
import matplotlib.pyplot as plt


from dash import dcc
import plotly.express as px

from . import logger, dash_app
from .analysis.base_analysis import BaseAnalysis
from .default.n_jobs import n_jobs

# %% ---- 2024-06-12 ------------------------
# Function and class


def compute_tfr_morlet(
    epochs, n_cycles: float = 4.0, segments: int = 16, h_freq: float = None
):
    # Compute the min frequency the epochs support
    # Ref: https://mne.tools/stable/generated/mne.time_frequency.tfr_morlet.html#mne.time_frequency.tfr_morlet
    freq_min = np.ceil(
        (5 / np.pi) / (len(epochs.times) + 1) * n_cycles * epochs.info["sfreq"]
    )

    if h_freq is not None:
        freq_max = h_freq
    else:
        freq_max = np.max([freq_min * 2, epochs.info["lowpass"]])

    assert (
        freq_max > freq_min
    ), f"freq max must be greater than min: {freq_max} > {freq_min}"

    freqs = np.linspace(freq_min, freq_max, segments)

    tfr_epochs = mne.time_frequency.tfr_morlet(
        epochs,
        freqs,
        picks=epochs.info.ch_names,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        n_jobs=n_jobs,
    )
    times = epochs.times
    tfr_epochs.apply_baseline(baseline=(times[0], 0))
    array = tfr_epochs.data.squeeze()
    averaged_array = tfr_epochs.average().data.squeeze()

    return tfr_epochs, freqs, times, array, averaged_array


class MI_Analysis(BaseAnalysis):
    protocol = 'MI'

    def __init__(self, protocol: str = None, files: list = None, options: dict = None):
        if files is None:
            files = []
        if options is None:
            options = {}
        super(MI_Analysis, self).__init__(protocol, files, options)
        self.load_methods()

    def load_methods(self):
        self.methods['Plot ERD'] = self.method_plot_erd

    def method_plot_erd(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs[selected_event_id]
        sfreq = epochs.info['sfreq']

        v_scale = 1e-10
        h_freq = np.min([sfreq/2, self.options['freqBand'].get('freq_h', 25)])

        n = len(self.options['channels'])
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

        for ax, sensor_name in zip(axes, self.options['channels']):
            tfr_epochs, freqs, times, array, averaged_array = compute_tfr_morlet(
                epochs.copy().pick([sensor_name]), h_freq=h_freq)
            evoked = tfr_epochs.average()
            evoked.plot(vmin=-v_scale, vmax=v_scale, axes=ax, show=False)
            title = f'Channel: {sensor_name}'
            ax.set_title(title)

            # Update dash_app
            _fig = px.imshow(
                evoked.data.squeeze(),
                y=freqs, origin='lower', x=epochs.times,
                aspect='auto',
                title=title, template="seaborn")
            dash_app.div.children.append(dcc.Graph(figure=_fig))

        title = f'TFR-morlet-evoked-{selected_event_id}'
        fig.suptitle(title)
        fig.tight_layout()

        return fig


# %% ---- 2024-06-12 ------------------------
# Play ground


# %% ---- 2024-06-12 ------------------------
# Pending


# %% ---- 2024-06-12 ------------------------
# Pending
