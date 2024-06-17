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
from PySide6.QtUiTools import QUiLoader

from .base_window import BaseWindow
from .base_protocol_window import BaseProtocolWindow
from . import logger, project_root, cache_path

# --------------------
loader = QUiLoader()
layout_path = project_root.joinpath('layout/MI.ui')

# %% ---- 2024-04-29 ------------------------
# Function and class


class MIDefaultOptions:
    """
    The attributes are type specific
    """
    channels = ['C3', 'CZ', 'C4']
    eventIds = ['200', '201', '202']
    epochTimes = dict(tmin=-1.0, tmax=5.0)
    freqBand = dict(freq_l=1.0, freq_h=25.0)
    rejects = dict(eeg=1e-6)
    epochsKwargs = dict(baseline=(None, 0), decim=10)


class MIWindow(BaseProtocolWindow):
    # --------------------
    # Known components
    # --
    plainTextEdit_eventIds = None
    plainTextEdit_epochTimes = None
    plainTextEdit_freqBand = None
    plainTextEdit_channels = None
    plainTextEdit_rejects = None
    plainTextEdit_epochsKwargs = None

    # --------------------
    # variables
    protocol = 'MI'

    def __init__(self, files: list, parent=None):
        # Initialize BaseProtocolWindow
        super().__init__(
            layout_path=layout_path,
            files=files,
            ClassOfDefaultOptions=MIDefaultOptions,
            parent=parent)

        self.bind_options_with_textEdits()
        self.load_default_operations()

        self.set_protocol_slogan('Motion imaging experiment')
        self._set_window_title('Motion imaging experiment')

        logger.info('Initialized MIWindow')

    def bind_options_with_textEdits(self):
        """
        Bind the options with their names
        It assign the textEdits for every options.

        ! Its keys should be exactly the same as the attrs of the MIDefaultOptions.
        """
        self.option_plainTextEdits = dict(
            eventIds=self.plainTextEdit_eventIds,
            epochTimes=self.plainTextEdit_epochTimes,
            freqBand=self.plainTextEdit_freqBand,
            channels=self.plainTextEdit_channels,
            rejects=self.plainTextEdit_rejects,
            epochsKwargs=self.plainTextEdit_epochsKwargs,
        )
        logger.debug(
            f'Set option plainTextEdits: {self.option_plainTextEdits}')


# %% ---- 2024-04-29 ------------------------
# Play ground


# %% ---- 2024-04-29 ------------------------
# Pending


# %% ---- 2024-04-29 ------------------------
# Pending
