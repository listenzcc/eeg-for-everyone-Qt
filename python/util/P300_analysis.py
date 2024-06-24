"""
File: P300_analysis.py
Author: Chuncheng Zhang
Date: 2024-06-17
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis for P300 Dataset

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-17 ------------------------
# Requirements and constants
import mne
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .analysis.base_analysis import BaseAnalysis
from .default.n_jobs import n_jobs
from . import logger

from .algorithm.BLDA.BLDA import BLDA, post_process_y_prob

# %% ---- 2024-06-17 ------------------------
# Function and class


class P300_Analysis(BaseAnalysis):
    protocol = 'P300'

    def __init__(self, protocol: str = None, files: list = None, options: dict = None):
        if files is None:
            files = []
        if options is None:
            options = {}
        super(P300_Analysis, self).__init__(protocol, files, options)
        self.load_methods()

    def load_methods(self):
        self.methods['debug'] = self.debug

    def debug(self, selected_idx, selected_event_id):
        epochs = self.objs[selected_idx].epochs[selected_event_id]
        epochs = self.objs[selected_idx].epochs
        sfreq = epochs.info['sfreq']

        # Data shape is (trials, channels, time-points)
        data = epochs.get_data()
        print(data.shape, sfreq)

        # Choose classification method for 'lda' (faster) or 'blda' (slower)
        method = 'lda'
        # method = 'blda'

        if method == 'blda':
            # ----------------------------------------
            # ---- Fit and predict with BLDA(Bei Wang) ----
            blda = BLDA(name='P300')

            # Transpose dim to fit (channels, time_series, stims)
            # Convert the >1 values to 1, 1 value to 0
            X = data.transpose((1, 2, 0))
            _y = np.array(epochs.events)[:, -1]
            y = _y.copy()
            y[_y > 1] = 1
            y[_y == 1] = 0
            y = y.reshape((len(y), 1))

            # Train & validation
            blda.fit(X, y)
            logger.debug(f'Trained with {X.shape}, {y.shape}')
            y_prob = blda.predict(X)
            y_pred = post_process_y_prob(y_prob)

        elif method == 'lda':
            # ----------------------------------------
            # ---- Fit and predict with LDA ----
            clf = LinearDiscriminantAnalysis()
            X = data.reshape(len(data), -1)
            y = np.array(epochs.events)[:, -1]

            # Train & validation
            clf.fit(X, y)
            y_prob = clf.predict_proba(X)[:, -1]
            y_pred = clf.predict(X)

        # ----------------------------------------
        # ---- Summary result ----
        c_mat = metrics.confusion_matrix(
            y_true=y, y_pred=y_pred, normalize='true')
        roc_auc_score = metrics.roc_auc_score(y_true=y, y_score=y_prob)
        print(y_pred.shape)
        print(c_mat)
        print(roc_auc_score)
        logger.debug(f'Predicted with {c_mat}, {roc_auc_score}')

        # ----------------------------------------
        # ---- Generate result figure ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.heatmap(c_mat, ax=ax)
        ax.set_title(f'Roc auc score is {roc_auc_score}')
        # time.sleep(5)
        return fig


# %% ---- 2024-06-17 ------------------------
# Play ground


# %% ---- 2024-06-17 ------------------------
# Pending


# %% ---- 2024-06-17 ------------------------
# Pending
