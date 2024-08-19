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
from io import StringIO
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dash import dash_table
from tqdm.auto import tqdm

from sklearn import metrics
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .analysis.base_analysis import BaseAnalysis
from .default.n_jobs import n_jobs
from .input_dialog import require_options_with_QDialog, require_file_path_with_QDialog
from . import logger, dash_app

from .algorithm.BLDA.BLDA import BLDA, post_process_y_prob
from .algorithm.EEGNet.EEGNet import EEGNet

# %% ---- 2024-06-17 ------------------------
# Function and class


def require_classifier_options(other_epochs):
    if other_epochs:
        default_options = {
            'method': 'lda',
        }
        logger.debug('Has other epochs')
    else:
        default_options = {
            'method': 'lda',
            'crossValidation': 5
        }
        logger.debug('Does not have other epochs')

    comment = '''
# The LDA requires the options
# - method = lda: Using sklearn.discriminant_analysis.LinearDiscriminantAnalysis for LDA discrimination
# - method = blda: Using Bei Wang's BLDA algorithm
# - method = EEGNet: Using EEGNet algorithm
'''

    return require_options_with_QDialog(default_options, comment, 'Require classifier name')


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
        self.methods['Classifier'] = self.Classifier

    def Classifier(self, selected_idx, selected_event_id, **kwargs):
        '''
        LDA, BLDA and EEGNet classification methods.
        '''
        # Select epochs of all the event_id
        # Pick given channels
        epochs = self.objs[selected_idx].epochs
        epochs = epochs.pick([e.upper() for e in self.options['channels']])

        # Collect other epochs
        other_epochs = [
            e.epochs.pick([e.upper() for e in self.options['channels']])
            for i, e in enumerate(self.objs) if i != selected_idx]

        # Get input options
        inp = require_classifier_options(other_epochs)

        # Choose classification method for 'lda' (faster) or 'blda' (slower)
        method = inp.get('method')
        cv = int(inp.get('crossValidation', 5))

        def _blda_get_X_y(epochs):
            # Convert the >1 values to 1, 1 value to 0
            _y = np.array(epochs.events)[:, -1]
            y = _y.copy()
            y[_y > 1] = 1
            y[_y == 1] = 0
            y = y.reshape((len(y), -1))

            # Data shape is (trials, channels, time-points)
            X = epochs.get_data(copy=True)
            return X, y

        def _lda_get_X_y(epochs):
            # Label y
            y = np.array(epochs.events)[:, -1]
            # Data shape is (trials, channels, time-points)
            X = epochs.get_data(copy=True)
            X = X.reshape(len(X), -1)
            return X, y

        # ----------------------------------------
        # ---- Train and test based on whether other_epochs are provided or not ----
        # If there are other_epochs, train on the other_epochs and test on the epochs
        # If there is not other_epochs, train and test on n_splits=cv folders validation

        if other_epochs and method.lower() == 'blda':
            # Clf
            clf = BLDA(name='P300')

            # Separate data
            pairs = [_blda_get_X_y(e) for e in other_epochs]
            train_X = np.concatenate([e[0] for e in pairs], axis=0)
            train_y = np.concatenate([e[1] for e in pairs], axis=0)
            test_X, test_y = _blda_get_X_y(epochs)
            logger.debug(
                f'Data shape: {train_X.shape}, {train_y.shape}, {test_X.shape}, {test_y.shape}')

            # Train & validation
            clf.fit(train_X.transpose((1, 2, 0)), train_y)
            y_prob = clf.predict(test_X.transpose((1, 2, 0)))
            y_pred = post_process_y_prob(y_prob)

        elif other_epochs and method.lower() == 'lda':
            # Clf
            clf = LinearDiscriminantAnalysis()

            # Separate data
            pairs = [_lda_get_X_y(e) for e in other_epochs]
            train_X = np.concatenate([e[0] for e in pairs], axis=0)
            train_y = np.concatenate([e[1] for e in pairs], axis=0)
            test_X, test_y = _lda_get_X_y(epochs)
            logger.debug(
                f'Data shape: {train_X.shape}, {train_y.shape}, {test_X.shape}, {test_y.shape}')

            # Train & validation
            '''
            模型初始化：
                请提供 Classifier 类，并详述初始化参数（如需要与输入数据耦合，请在这部分给出）

                clf = Classifier(**kwargs)
                
                - Classifier 至少需要提供这些方法：
                    - clf.fit(X, y), 模型训练
                    - clf.predict_proba(X), 模型测试，输出为样本对应的各个类别的概率或得分
                    - clf.predict(X), 模型测试，输出为样本对应的类别

            使用时的输入变量：
                - train_X (np.ndarray), 训练数据，请规定它的维度，以及各个维度与eeg数据的对应关系
                - train_y (np.ndarray), 训练数据标签，请规定它的维度，以及与train_X的对应关系
                - test_X (np.ndarray), 测试数据，请规定它的维度，以及各个维度与eeg数据的对应关系
                - test_y (np.ndarray), 训练数据标签，请规定它的维度，以及与test_X的对应关系
            
            使用时的输出变量：
                - y_pred (np.ndarray), 测试数据经过clf.predict_proba得到的预测score，请规定它的维度
            '''
            clf.fit(train_X, train_y)
            y_prob = clf.predict_proba(test_X)
            y_prob = y_prob[:, -1]
            y_pred = clf.predict(test_X)

        elif not other_epochs and method.lower() == 'blda':
            # ----------------------------------------
            # ---- Fit and predict with BLDA(Bei Wang) ----
            clf = BLDA(name='P300')

            # Get X, y
            test_X, test_y = _blda_get_X_y(epochs)

            # Train & validation
            skf = model_selection.StratifiedKFold(n_splits=cv)
            n = len(test_y)
            y_prob = np.zeros((n, 1))
            for i, (train_index, test_index) in tqdm(enumerate(skf.split(test_X, test_y))):
                # Transpose dim to fit (channels, time_series, trials)
                clf.fit(test_X[train_index].transpose(
                    (1, 2, 0)), test_y[train_index])
                y_prob[test_index] = clf.predict(
                    test_X[test_index].transpose((1, 2, 0)))
            y_pred = post_process_y_prob(y_prob)

        elif not other_epochs and method.lower() == 'lda':
            # ----------------------------------------
            # ---- Fit and predict with LDA ----
            # Train & validation
            clf = LinearDiscriminantAnalysis()

            # Get X, y
            test_X, test_y = _lda_get_X_y(epochs)

            # Train & validation
            y_prob = model_selection.cross_val_predict(
                clf, test_X, test_y, cv=cv, n_jobs=n_jobs, method='predict_proba')
            y_prob = y_prob[:, -1]
            y_pred = model_selection.cross_val_predict(
                clf, test_X, test_y, cv=cv, n_jobs=n_jobs, method='predict')

        elif method.lower() == 'eegnet':
            # ----------------------------------------
            # ---- Fit and predict with EEGNet ----

            # Get net
            net = EEGNet(model_path=require_file_path_with_QDialog())

            # Fit the network
            # TODO: Actually train the model. Now the method is bypassed since the pre-trained model is used.
            net.trained = True

            # Get X, y
            test_X, test_y = _blda_get_X_y(epochs)

            # Predict
            print(test_X.shape, test_y.shape)
            y_prob = net.predict_proba(test_X)
            y_pred = net.predict(test_X)

        else:
            msg = f'Unknown method: {method}, or other things are incorrect.'
            logger.error(msg)
            assert False, msg

        # ----------------------------------------
        # ---- Summary result ----
        report = metrics.classification_report(
            y_true=test_y, y_pred=y_pred, output_dict=True)
        c_mat = metrics.confusion_matrix(
            y_true=test_y, y_pred=y_pred, normalize='true')
        roc_auc_score = metrics.roc_auc_score(y_true=test_y, y_score=y_prob)
        logger.debug(f'Prediction result: {report}, {c_mat}, {roc_auc_score}')

        # ----------------------------------------
        # ---- Generate result figure ----
        # require_options_with_QDialog_thread(
        #     default_options=report,
        #     comment='# Classification result')

        si = StringIO(json.dumps(report))
        df = pd.read_json(si)
        columns = df.columns
        df['measurement'] = df.index
        columns = columns.insert(0, 'measurement')
        df = df[columns]
        print(df)

        dash_app.div.children.append(dash_table.DataTable(
            df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
            filter_action="native",
            filter_options={"placeholder_text": "Filter column..."},
            page_size=10,
        ))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.heatmap(c_mat, ax=ax, annot=c_mat)
        ax.set_xlabel('True label')
        ax.set_ylabel('Predict label')
        ax.set_title(
            f'Method: {method}, Roc auc score is {roc_auc_score:0.2f}')
        return fig


# %% ---- 2024-06-17 ------------------------
# Play ground


# %% ---- 2024-06-17 ------------------------
# Pending


# %% ---- 2024-06-17 ------------------------
# Pending
