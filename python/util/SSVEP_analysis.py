"""
File: SSVEP_analysis.py
Author: Chuncheng Zhang
Date: 2024-06-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis for SSVEP Dataset

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-20 ------------------------
# Requirements and constants
import mne
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from PySide6 import QtWidgets

from dash import dash_table
from scipy import signal
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .analysis.base_analysis import BaseAnalysis
from .input_dialog import require_options_with_QDialog
from .default.n_jobs import n_jobs
from . import logger, dash_app

from .algorithm.FBCCA_code.FBCCA import FBCCA
from .algorithm.FBCCA_code.CCA import CCA

# %% ---- 2024-06-20 ------------------------
# Function and class


def mk_template(
        sfreq: int = None,
        candidate_freqs: list = None,
        multiplier_factor: int = None,
        trial_length: int = None):
    """
    构建正余弦模板.

    Args:
        sfreq: 输入采样率(Hz).
        candidate_freqs: 刺激频率(Hz), 倍频数量, 刺激时长(秒) -> time points.
        multiplier_factor: 倍频系数.
        trial_length: 每个trial的数据长度.

    Returns:
        List of target template sets.
    """

    target_template_set = []
    # cal_len = stim_time * samp_rate
    samp_point = np.linspace(
        0, (trial_length - 1) / sfreq, trial_length, endpoint=True)
    # (1 * 计算长度)的二维矩阵
    samp_point = samp_point.reshape(1, len(samp_point))
    for freq in candidate_freqs:
        # 基频 + 倍频
        test_freq = np.linspace(
            freq, freq * multiplier_factor, multiplier_factor, endpoint=True
        )
        # (1 * 倍频数量)的二维矩阵
        test_freq = test_freq.reshape(1, len(test_freq))
        # (倍频数量 * 计算长度)的二维矩阵
        num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
        cos_set = np.cos(num_matrix)
        sin_set = np.sin(num_matrix)
        cs_set = np.append(cos_set, sin_set, axis=0)
        target_template_set.append(cs_set)
    return target_template_set


def mk_pre_notch_filter(sfreq: int, f0: float = 50.0, q: int = 20):
    # q: 凝胶电极需要稍微降低品质因数
    b, a = signal.iircomb(f0, q, ftype='notch', fs=sfreq)
    return dict(b=b, a=a)


def mk_filter_bank(sfreq, low_pass_band, high_pass_band):
    print(sfreq, low_pass_band, high_pass_band)

    # SSVEP
    fs = sfreq / 2
    A = []
    B = []
    for fl, fh in zip(low_pass_band, high_pass_band):
        N, Wn = signal.ellipord(
            [fl / fs, fh / fs], [(fl - 4) / fs, (fh + 10) / fs], 3, 40)
        b, a = signal.ellip(N, 1, 40, Wn, 'bandpass')
        A.append(a)
        B.append(b)
    return dict(B=B, A=A)


def require_FBCCA_options(event_id: dict):
    # Default options
    def convert_v_to_freq(v):
        # Convert 1-40 to 8-15.8 Hz
        return round(8.0 + v * 0.2, 1)

    default_options = {
        'a': -1.25,
        'b': 0.25,
        'multiplierFactor': 5,
        'iirF0': 50.0,
        'iirQuality': 20,
        'nFilterBank': 7,
        'lowPassBand': [6, 14, 22, 30, 38, 46, 54],
        'highPassBand': [90, 90, 90, 90, 90, 90, 90],
        'eventId': event_id,
        'frequencies': {k: convert_v_to_freq(v) for k, v in event_id.items()},
    }

    comment = '''
# The FBCCA method requires the options:
'''

    return require_options_with_QDialog(default_options, comment)


class SSVEP_Analysis(BaseAnalysis):
    protocol = 'SSVEP'

    def __init__(self, protocol: str = None, files: list = None, options: dict = None):
        if files is None:
            files = []
        if options is None:
            options = {}
        super(SSVEP_Analysis, self).__init__(protocol, files, options)
        self.load_methods()

    def load_methods(self):
        self.methods['FBCCA'] = self.FBCCA

    def FBCCA(self, selected_idx, selected_event_id, **kwargs):
        '''
        The FBCCA classification method for SSVEP dataset.
        The method is a no-train method.

        Args:
            - selected_idx (int): The selected file index.
            - selected_event_id (int): The selected event id.
            - **kwargs: The kwargs placeholder for compatibility with other methods. (It is not used in this method).
        '''
        # Select epochs of all the event_id
        epochs = self.objs[selected_idx].epochs

        # Pick given channels
        epochs = epochs.pick([e.upper() for e in self.options['channels']])

        # ----------------------------------------
        # ---- User input for frequencies ----
        inp = require_FBCCA_options(epochs.event_id)
        logger.debug(f'Got input: {inp}')

        # ----------------------------------------
        # ---- Prepare data & options ----
        # Required options
        sfreq = epochs.info['sfreq']
        multiplier_factor = int(inp.get('multiplierFactor'))
        low_pass_band = inp.get('lowPassBand')
        high_pass_band = inp.get('highPassBand')
        a = float(inp.get('a'))
        b = float(inp.get('b'))
        f0 = float(inp.get('iirF0'))
        q = int(inp.get('iirQuality'))
        nFB = np.min((len(low_pass_band), len(high_pass_band)))
        candidate_freqs = [float(v) for k, v in inp['frequencies'].items()]

        # Compute filter bank
        filter_bank = mk_filter_bank(sfreq, low_pass_band, high_pass_band)

        # Data epochs shape is (trials, channels, time-points)
        data_epochs = epochs.get_data()
        trial_length = data_epochs.shape[2]

        target_template_set = mk_template(
            sfreq, candidate_freqs, multiplier_factor, trial_length)

        # Make pre notch filter
        pre_notch_filter = mk_pre_notch_filter(sfreq, f0=f0, q=q)

        # FBCCA
        fbcca = FBCCA(target_template_set, a, b, nFB)

        # Predict for each trial data
        preds = []
        for data in tqdm(data_epochs, 'Predicting'):
            # Protect data
            data = data.copy()

            # Pre notch
            data = signal.filtfilt(
                pre_notch_filter['b'], pre_notch_filter['a'], data, axis=1)

            # Filter bank
            X = np.array([
                signal.filtfilt(b, a, data, axis=1)
                for b, a in zip(filter_bank['B'], filter_bank['A'])])
            pred = fbcca.predict(X)
            preds.append((pred, candidate_freqs[pred]))

        result = []
        event_id_inv = {v: k for k, v in epochs.event_id.items()}
        for pred, event in zip(preds, epochs.events):
            res = dict(
                pred_event=pred[0],
                pred_freq=pred[1],
                true_event=event[2],
                true_freq=inp['frequencies'].get(event_id_inv[event[2]]),
                event_id=event_id_inv[event[2]]
            )
            result.append(res)
        df = pd.DataFrame(result)
        df['error'] = df['pred_freq'] - df['true_freq']

        # Display the prediction results
        print(df)

        # Compute and display frequency response
        if kwargs.get('flag_require_detail', True):
            logger.debug('Require detail')
            self._FBCCA_plot_filter(pre_notch_filter, filter_bank, sfreq)

        # ----------------------------------------
        # ---- Generate result figure ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.regplot(df, x='true_freq', y='pred_freq', ax=ax)
        sns.scatterplot(
            df, x='true_freq', y='pred_freq', s=100, hue='error', ax=ax)
        for _, se in df.iterrows():
            plt.text(
                se['true_freq'], se['pred_freq'], f'*{se["event_id"]}', horizontalalignment='left')

        # ----------------------------------------
        # ---- Update dash_app children ----
        _df = df.copy()
        for col in _df.columns:
            _df[col] = _df[col].map(str)

        dash_app.div.children.append(dash_table.DataTable(
            _df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
            filter_action="native",
            filter_options={"placeholder_text": "Filter column..."},
            page_size=10,
        ))

        # Plot selected_event_id in 'red' color
        selected_df = df.query(f'event_id=="{selected_event_id}"')
        sns.scatterplot(
            selected_df, x='true_freq', y='pred_freq', color='red', s=200, ax=ax)
        for _, se in selected_df.iterrows():
            plt.text(
                se['true_freq'], se['pred_freq'], f'*{selected_event_id} | {se["pred_freq"]} | {se["true_freq"]}', horizontalalignment='left')

        ax.grid(True)
        return fig

    def _FBCCA_plot_filter(self, pre_notch_filter, filter_bank, sfreq):
        # ----------------------------------------
        # ---- Notch filter ----
        w, h = signal.freqz(pre_notch_filter['b'], pre_notch_filter['a'])
        w /= np.pi
        w *= sfreq / 2
        fig, axs = plt.subplots(2, 2, figsize=(8, 4))

        ax = axs[0][0]
        ax.semilogx(w, 20*np.log10(h))
        ax.set_xlabel('Frequency')
        ax.set_ylabel('dB')
        ax.set_title('Notch Filter Frequency Response (Magnitude)')
        ax.grid(True)

        ax = axs[1][0]
        ax.semilogx(w, np.unwrap(np.angle(h)))
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Phase(rad)')
        ax.grid(True)

        # ----------------------------------------
        # ---- Filter bank filter ----
        for b, a in zip(filter_bank['B'], filter_bank['A']):
            w, h = signal.freqz(b, a)
            w /= np.pi
            w *= sfreq / 2

            ax = axs[0][1]
            ax.semilogx(w, 20*np.log10(h))

            ax = axs[1][1]
            ax.semilogx(w, np.unwrap(np.angle(h)))

        ax = axs[0][1]
        ax.set_ylim([-100, 10])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('dB')
        ax.set_title('Bank Filter Frequency Response (Magnitude)')
        ax.grid(True)

        ax = axs[1][1]
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Phase(rad)')
        ax.grid(True)

        fig.tight_layout()
        plt.show()

        logger.debug('Plotted filter details of FBCCA')

# %% ---- 2024-06-20 ------------------------
# Play ground


# %% ---- 2024-06-20 ------------------------
# Pending


# %% ---- 2024-06-20 ------------------------
# Pending
