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

from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dash import dcc
import plotly.express as px

from . import logger, dash_app
from .analysis.base_analysis import BaseAnalysis
from .algorithm.MIShallowCNN.shallow import shallow_cnn_decoding
from .default.n_jobs import n_jobs

# %% ---- 2024-06-12 ------------------------
# Function and class


def fbcsp_decoding_with_confusion_matrix(epochs, events, n_folds=5, bands=[(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36)]):
    """
    FBCSP解码方法实现，返回准确率和混淆矩阵

    参数:
        epochs: mne.Epochs对象
        events: 事件标签数组
        n_folds: 交叉验证折数 (默认5)
        bands: 频带列表 (默认8个频带)

    返回:
        mean_accuracy: 平均准确率
        all_accuracies: 每折的准确率
        fig: 混淆矩阵的matplotlib Figure对象
    """
    X = epochs.get_data()
    y = events[:, -1]
    unique_classes = np.unique(y)

    # 初始化交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    all_accuracies = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(np.unique(y_train), np.unique(y_test))

        train_features = []
        test_features = []

        for fmin, fmax in bands:
            # 频带滤波
            X_train_filt = filter_data(
                X_train, epochs.info['sfreq'], fmin, fmax, verbose=False)
            X_test_filt = filter_data(
                X_test, epochs.info['sfreq'], fmin, fmax, verbose=False)

            # CSP特征提取
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            csp.fit(X_train_filt, y_train)

            # 转换数据
            train_features.append(csp.transform(X_train_filt))
            test_features.append(csp.transform(X_test_filt))

        # 合并特征
        X_train_final = np.concatenate(train_features, axis=1)
        X_test_final = np.concatenate(test_features, axis=1)

        # LDA分类
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_final, y_train)
        y_pred = lda.predict(X_test_final)

        # 存储结果
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        acc = accuracy_score(y_test, y_pred)
        all_accuracies.append(acc)
        print(f"Fold accuracy: {acc:.3f}")

    # 计算平均准确率
    mean_accuracy = np.mean(all_accuracies)
    print(f"\nMean accuracy: {mean_accuracy:.3f}")

    # 创建混淆矩阵图形
    cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'FBCSP Confusion Matrix (Mean Acc: {mean_accuracy:.3f})')
    plt.tight_layout()

    return mean_accuracy, all_accuracies, fig


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
        self.methods['FBCSP'] = self.method_fbscp_decoding
        self.methods['ShallowCNN'] = self.method_shallow_cnn_decoding

    def method_shallow_cnn_decoding(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs
        mean_accuracy, all_accuracies, fig = shallow_cnn_decoding(
            epochs, epochs.events)

        title = 'ShallowCNN Decoding'
        fig.suptitle(title)
        fig.tight_layout()
        self.append_report_fig(fig, 'ShallowCNN', selected_idx)
        return fig

    def method_fbscp_decoding(self, selected_idx, selected_event_id, **kwargs):
        epochs = self.objs[selected_idx].epochs
        mean_accuracy, all_accuracies, fig = fbcsp_decoding_with_confusion_matrix(
            epochs, epochs.events)

        title = 'FBCSP Decoding'
        fig.suptitle(title)
        fig.tight_layout()
        self.append_report_fig(fig, 'FBCSP', selected_idx)
        return fig

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
        self.append_report_fig(fig, 'ERD', selected_idx, selected_event_id)
        return fig


# %% ---- 2024-06-12 ------------------------
# Play ground


# %% ---- 2024-06-12 ------------------------
# Pending


# %% ---- 2024-06-12 ------------------------
# Pending
