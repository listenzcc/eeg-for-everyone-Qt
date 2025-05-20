"""
File: window_of_MI.py
Author: Chuncheng Zhang
Date: 2024-04-29
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


# %% ---- 2024-04-29 ------------------------
# Requirements and constants
import time
import random
from threading import Thread
from datetime import datetime

from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

from .base.base_protocol_window import BaseProtocolWindow
from .window_of_analysis_results import AnalysisResultsWindow

from . import default_options
from . import MI_Analysis, P300_Analysis, SSVEP_Analysis, BaseAnalysis
from . import logger, project_root, cache_path

# --------------------
loader = QUiLoader()
layout_path = project_root.joinpath('layout/setup_options.ui')

# %% ---- 2024-04-29 ------------------------
# Function and class

# Load preprocessing options for each experiment
default_options_candidates = {
    'MI': default_options.MIDefaultOptions,
    'P300(3X3)': default_options.P300DefaultOptions,
    'P300(二项式)': default_options.P300DefaultOptions,
    'SSVEP': default_options.SSVEPDefaultOptions,
    'default': default_options.AnyDefaultOptions
}

# Load analysis object for each experiment
analysis_candidates = {
    'MI': MI_Analysis,
    'P300(3X3)': P300_Analysis,
    'P300(二项式)': P300_Analysis,
    'SSVEP': SSVEP_Analysis,
    'default': BaseAnalysis,
}


class SetupOptionsWindow(BaseProtocolWindow):
    # --------------------
    # Known components
    # --
    plainTextEdit_eventIds = None
    plainTextEdit_epochTimes = None
    plainTextEdit_freqBand = None
    plainTextEdit_channels = None
    plainTextEdit_reject = None
    plainTextEdit_epochsKwargs = None
    plainTextEdit_otherOptions = None
    progressBar: QtWidgets.QProgressBar = None
    buttonBox_goToNext: QtWidgets.QDialogButtonBox = None
    textBrowser_tracebackMessage: QtWidgets.QListWidget = None

    # --------------------
    # variables
    protocol = None

    def __init__(self, files: list, protocol: str, parent=None):
        # Initialize BaseProtocolWindow

        # Bind the option class
        class_of_default_options = default_options_candidates.get(
            protocol, default_options_candidates['default'])

        super().__init__(
            layout_path=layout_path,
            files=files,
            ClassOfDefaultOptions=class_of_default_options,
            parent=parent)

        self.protocol = protocol

        self.bind_options_with_textEdits()
        self.load_default_operations()

        title = f'{class_of_default_options.short_name}: {
            class_of_default_options.long_name}'
        self.set_protocol_slogan(title)
        self._set_window_title(title)

        self.handle_goToNext_events()

        self.progressBar.setValue(0)
        logger.info(f'Initialized with protocol {protocol}')

    def bind_options_with_textEdits(self):
        """
        Bind the options with their names
        It assign the textEdits for every options.

        ! Its keys are exactly the same as the attrs of the defaultOptions.
        """

        # It is the table of the options and keys.
        # If the right side is changed, the left side will be changed accordingly.
        self.option_plainTextEdits = dict(
            eventIds=self.plainTextEdit_eventIds,
            epochTimes=self.plainTextEdit_epochTimes,
            freqBand=self.plainTextEdit_freqBand,
            channels=self.plainTextEdit_channels,
            reject=self.plainTextEdit_reject,
            epochsKwargs=self.plainTextEdit_epochsKwargs,
            otherOptions=self.plainTextEdit_otherOptions
        )
        logger.debug(
            f'Set option plainTextEdits: {self.option_plainTextEdits}')

    def handle_goToNext_events(self):
        def _accept():
            files = self.chosen_files
            protocol = self.protocol
            options = self.options

            # Correct the channels into ['eeg'] or uppercase channel names.
            if channels := options.get('channels'):
                if channels == ['eeg']:
                    pass
                else:
                    options['channels'] = [e.upper() for e in channels]

            traceback_message = [
                f'Analysis of {protocol} started at {datetime.now()}',
                '!!! No news is good news']

            def report_traceback():
                self.textBrowser_tracebackMessage.setText(
                    '\n\n'.join(traceback_message))
                self.textBrowser_tracebackMessage.repaint()

            thread = Thread(target=self._progress_bar_engage,
                            daemon=True)
            thread.start()
            self.buttonBox_goToNext.setEnabled(False)

            try:
                # ----------------------------------------
                # ---- Start analysis ----
                report_traceback()
                tic = time.time()

                _analysis_cls: BaseAnalysis = analysis_candidates.get(
                    protocol, analysis_candidates['default'])
                current_analysis = _analysis_cls(protocol, files, options)

                costs = time.time() - tic
                traceback_message.append(
                    f'Analysis of {protocol} finished at {datetime.now()}, cost {costs} seconds')
                report_traceback()

                # ----------------------------------------
                # ---- Report brief ----
                n = len(current_analysis.objs)
                for i, obj in enumerate(current_analysis.objs):
                    traceback_message.append(
                        f'---- Epochs {i+1} | {n} ----\n{obj.epochs}\n{obj.epochs.info}')
                report_traceback()

                # ----------------------------------------
                # ---- Start results window ----
                window = AnalysisResultsWindow(current_analysis, self.window)
                window.show()
                self.textBrowser_tracebackMessage.setStyleSheet(
                    'background-color: #bfffbf')

            except Exception as error:
                import traceback
                exc = traceback.format_exc()
                print(exc)
                logger.error(f'{protocol} raises error: {error}')
                traceback_message.append(
                    f'Analysis of {protocol} got error: {error} at {datetime.now()}')
                traceback_message.append(exc)
                report_traceback()
                self.textBrowser_tracebackMessage.setStyleSheet(
                    'background-color: #fedfe1')

            finally:
                self._progress_bar_going_flag = False
                thread.join()
                self.buttonBox_goToNext.setEnabled(True)
                logger.debug(
                    f'Accepted options: {protocol} | {options} | {files}')

        self.buttonBox_goToNext.accepted.disconnect()
        self.buttonBox_goToNext.accepted.connect(_accept)

    def _progress_bar_engage(self):
        self._progress_bar_going_flag = True

        t = 0
        while self._progress_bar_going_flag:
            t += random.randint(1, 5)
            t %= 100
            self.progressBar.setValue(t)
            time.sleep(random.random())

        self.progressBar.setValue(100)
        logger.debug('Progress bar finished')

# %% ---- 2024-04-29 ------------------------
# Play ground

# %% ---- 2024-04-29 ------------------------
# Pending

# %% ---- 2024-04-29 ------------------------
# Pending
