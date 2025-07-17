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

from dash import dash_table, dcc
from tqdm.auto import tqdm

import plotly.express as px

from sklearn import metrics
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .analysis.base_analysis import BaseAnalysis
from .analysis.compute_mutual_information import mutual_information_from_confusion_matrix

from .default.n_jobs import n_jobs
from .input_dialog import require_options_with_QDialog, require_file_path_with_QDialog
from . import logger, dash_app

from .algorithm.BLDA.BLDA import BLDA, post_process_y_prob
from .algorithm.EEGNet.EEGNet import EEGNet
from .algorithm.R_Sqrt import r2_sqrt
from .algorithm.Lasso_Preprocess import Lasso_Process
from .algorithm.Preprocess_ERP.main import EEGProcessor

# %% ---- 2024-06-17 ------------------------
# Function and class


def require_classifier_options(other_epochs):
    if other_epochs:
        default_options = {
            'method': 'lda',
            'lassoFlag': 'True',
            'lassoAlpha': 0.01
        }
        logger.debug('Has other epochs')
    else:
        default_options = {
            'method': 'lda',
            'crossValidation': 5,
            'lassoFlag': 'True',
            'lassoAlpha': 0.01
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
        self.methods['R2Index'] = self.r2_index

    def r2_index(self, selected_idx, selected_event_id, **kwargs):
        # Load the epochs and read the data & events
        epochs = self.objs[selected_idx].epochs

        def convert_events(events):
            # Convert 1 to 0, larger numbers to 1
            output = np.array(events).copy()
            output[events == 1] = 0
            output[events > 1] = 1
            return output

        events = convert_events(epochs.events[:, -1])
        data = epochs.get_data()
        r2_matrix = r2_sqrt(data, events)

        # Render html

        kwargs = dict(
            title='R2 index', x=epochs.times,
            y=epochs.info['ch_names'])
        fig = px.imshow(r2_matrix, aspect='auto', **kwargs)
        dash_app.div.children.append(dcc.Graph(figure=fig))

        # Plot
        evoked = epochs.average()
        # Convert the ruler to -1 ~ 1 by force
        evoked.data = r2_matrix * 1e-6

        fig = evoked.plot_joint(
            title='R2 index',
            show=False,
            exclude=['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'],
            ts_args=dict(units={'eeg': 'R2'})
        )

        self.append_report_fig(fig, 'R2Index', selected_idx, selected_event_id)

        return fig

    def Classifier(self, selected_idx, selected_event_id, **kwargs):
        '''
        LDA, BLDA and EEGNet classification methods.
        '''
        # Select epochs of all the event_id
        # Pick given channels
        epochs = self.objs[selected_idx].epochs
        epochs = epochs.pick([e.upper() for e in self.options['channels']])
        print(epochs.get_data().shape)

        # Collect other epochs
        other_epochs = [
            e.epochs.pick([e.upper() for e in self.options['channels']])
            for i, e in enumerate(self.objs) if i != selected_idx]

        # Get input options
        inp = require_classifier_options(other_epochs)
        logger.debug(f'Got input options: {inp}')

        # Choose classification method for 'lda' (faster) or 'blda' (slower)
        method = inp.get('method')
        cv = int(inp.get('crossValidation', 5))
        lasso_flag = inp.get('lassoFlag', '').lower() == 'true'
        lasso_alpha = float(inp.get('alpha', 0.01))
        if lasso_flag:
            logger.debug(f'Using lasso and its alpha is {lasso_alpha}')

        def _blda_get_X_y(epochs):
            # Convert the >1 values to 1, 1 value to 0
            _y = np.array(epochs.events)[:, -1]
            y = _y.copy()
            y[_y > 1] = 1
            y[_y == 1] = 0
            y = y.reshape((len(y), -1))

            # Data shape is (trials, channels, time-points)
            X = epochs.get_data(copy=True)
            logger.debug(f'Got X({X.shape}) and y({y.shape}) for blda.')
            return X, y

        def _lda_get_X_y(epochs):
            # Label y
            y = np.array(epochs.events)[:, -1]
            # Data shape is (trials, channels, time-points)
            X = epochs.get_data(copy=True)
            # X = X.reshape(len(X), -1)
            logger.debug(f'Got X({X.shape}) and y({y.shape}) for lda.')
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

            # ----------------------------------------
            # ---- Lasso selection ----
            if lasso_flag:
                lp = Lasso_Process(alpha=lasso_alpha)
                selected_channels = lp.select_channels_index(train_X, train_y)
                logger.debug(
                    f'Lasso selects {len(selected_channels)} channels')

                # Select all the channels if the lasso selects nothing
                if len(selected_channels) == 0:
                    selected_channels = list(range(test_X.shape[1]))
                    logger.warning(
                        'Using all the channels since the lasso selects nothing')
            else:
                # Select all the channels if not using lasso
                selected_channels = list(range(test_X.shape[1]))

            train_X = train_X[:, selected_channels]
            test_X = test_X[:, selected_channels]

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

            # ----------------------------------------
            # ---- Lasso selection ----
            if lasso_flag:
                lp = Lasso_Process(alpha=lasso_alpha)
                selected_channels = lp.select_channels_index(train_X, train_y)
                logger.debug(
                    f'Lasso selects {len(selected_channels)} channels')

                # Select all the channels if the lasso selects nothing
                if len(selected_channels) == 0:
                    selected_channels = list(range(test_X.shape[1]))
                    logger.warning(
                        'Using all the channels since the lasso selects nothing')
            else:
                # Select all the channels if not using lasso
                selected_channels = list(range(test_X.shape[1]))

            # Flatten the data dimensions
            train_X = train_X[:, selected_channels].reshape(len(train_X), -1)
            test_X = test_X[:, selected_channels].reshape(len(test_X), -1)

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
            for i, (train_index, test_index) in tqdm(list(enumerate(skf.split(test_X, test_y)))):
                if lasso_flag:
                    lp = Lasso_Process(alpha=lasso_alpha)
                    selected_channels = lp.select_channels_index(
                        test_X[train_index], test_y[train_index])
                    logger.debug(
                        f'Lasso selects {len(selected_channels)} channels')

                    # Select all the channels if the lasso selects nothing
                    if len(selected_channels) == 0:
                        selected_channels = list(range(test_X.shape[1]))
                        logger.warning(
                            'Using all the channels since the lasso selects nothing')
                else:
                    # Select all the channels if not using lasso
                    selected_channels = list(range(test_X.shape[1]))

                # Transpose dim to fit (channels, time_series, trials)
                a = test_X[train_index][:,
                                        selected_channels].transpose((1, 2, 0))
                b = test_y[train_index]
                c = test_X[test_index][:,
                                       selected_channels].transpose((1, 2, 0))

                # Train & test
                clf.fit(a, b)
                y_prob[test_index] = clf.predict(c)

                # # Transpose dim to fit (channels, time_series, trials)
                # clf.fit(test_X[train_index].transpose(
                #     (1, 2, 0)), test_y[train_index])

                # y_prob[test_index] = clf.predict(
                #     test_X[test_index].transpose((1, 2, 0)))

            y_pred = post_process_y_prob(y_prob)

        elif not other_epochs and method.lower() == 'lda':
            # ----------------------------------------
            # ---- Fit and predict with LDA ----
            # Train & validation
            clf = LinearDiscriminantAnalysis()

            # Get X, y
            test_X, test_y = _lda_get_X_y(epochs)

            # Train & validation
            skf = model_selection.StratifiedKFold(n_splits=cv)
            n = len(test_y)
            y_pred = np.zeros((n, 1))
            y_prob = np.zeros((n, 1))
            for i, (train_index, test_index) in tqdm(list(enumerate(skf.split(test_X, test_y)))):
                if lasso_flag:
                    lp = Lasso_Process(alpha=lasso_alpha)
                    selected_channels = lp.select_channels_index(
                        test_X[train_index], test_y[train_index])
                    logger.debug(
                        f'Lasso selects {len(selected_channels)} channels')

                    # Select all the channels if the lasso selects nothing
                    if len(selected_channels) == 0:
                        selected_channels = list(range(test_X.shape[1]))
                        logger.warning(
                            'Using all the channels since the lasso selects nothing')
                else:
                    # Select all the channels if not using lasso
                    selected_channels = list(range(test_X.shape[1]))

                a = test_X[train_index][:, selected_channels].reshape(
                    len(train_index), -1)
                b = test_y[train_index]
                c = test_X[test_index][:, selected_channels].reshape(
                    len(test_index), -1)

                clf.fit(a, b)
                y_pred[test_index, 0] = clf.predict(c)
                # I need the proba of the largest label value
                y_prob[test_index, 0] = clf.predict_proba(c)[:, -1]

        elif method.lower() == 'eegnet':
            # ----------------------------------------
            # ---- Fit and predict with EEGNet ----

            # Get net
            directory = self.files[0]['path'].parent.as_posix()
            net = EEGNet(MODEL_PATH=require_file_path_with_QDialog(
                directory=directory))

            # ! Add by ZhenChen
            print(directory)
            parameter = {'PATH': directory,
                         'SELECT_CHANNEL': ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F3',
                                            'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7', 'FT8', 'Cz',
                                            'C3', 'C4', 'T7', 'T8', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz',
                                            'P3', 'P4', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO7', 'PO8',
                                            'Oz', 'O1', 'O2']}

            pro = EEGProcessor(**parameter)
            data, events, channels_name, correct = pro.run()

            # Now the method is bypassed since the pre-trained model is used.
            net.trained = True

            # Get X, y
            # test_X, test_y = _blda_get_X_y(epochs)
            test_X, test_y = data, events[:, 1]

            # Predict
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

        # roc_auc_score = metrics.roc_auc_score(
        #     y_true=test_y, y_score=y_prob, multi_class='ovr')
        acc_score = metrics.accuracy_score(
            y_true=test_y, y_pred=y_pred, normalize=True)
        logger.debug(f'Prediction result: {report}, {c_mat}, {acc_score}')

        # Compute Mutual Information
        confusion_matrix = metrics.confusion_matrix(
            y_true=test_y, y_pred=y_pred)
        m_i = mutual_information_from_confusion_matrix(confusion_matrix)
        t = np.max(epochs.times)
        itr = m_i / t
        if 'i_need_ITR' in kwargs:
            setattr(kwargs['i_need_ITR'], 'ITR', itr)
        logger.debug(
            f'The mutual information is {m_i} bits, time is {t} seconds, itr is {itr} bits/seconds')

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
            f'Method: {method}, Acc score is {acc_score:0.2f}')

        self.append_report_fig(
            fig, f'Classifier({method})', selected_idx, selected_event_id)
        return fig


# %% ---- 2024-06-17 ------------------------
# Play ground


# %% ---- 2024-06-17 ------------------------
# Pending


# %% ---- 2024-06-17 ------------------------
# Pending
