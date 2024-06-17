"""
File: base_protocol_window.py
Author: Chuncheng Zhang
Date: 2024-05-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The base class of protocol window.
    The protocol window is the analysis page for specific protocol (EEG experiment protocol).

    The purpose of the window is to handle the analysis options.
    ? And display the results (not sure yet)

    ! At least, the UI should have the required components.
    ! And the BaseProtocolWindow handles them appropriately.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-05-09 ------------------------
# Requirements and constants
from pathlib import Path
from PySide6.QtUiTools import QUiLoader

from .base_window import BaseWindow
from . import logger

loader = QUiLoader()

# %% ---- 2024-05-09 ------------------------
# Function and class


class BaseProtocolWindow(BaseWindow):
    # --------------------
    # Known components
    label_protocolSlogan = None
    listWidget_chosenFiles = None
    listWidget_analysisOptions = None
    pushButton_useDefault = None

    # --------------------
    # Variables
    ClassOfDefaultOptions = None  # The class of the default options
    chosen_files = []
    option_plainTextEdits = {}
    options = {}
    layout_path = None

    def __init__(self, layout_path: Path, files: list, ClassOfDefaultOptions, parent):
        self.layout_path = layout_path
        super().__init__(loader.load(layout_path, parent))
        self._load_files(files)
        self._load_default_option_class(ClassOfDefaultOptions)
        self._handle_useDefault_event()
        logger.info(f'Initialized with {layout_path}')

    def _load_files(self, files: list):
        """
        Loads the specified files into the protocol window.

        Args:
            files (list): A list of files to be loaded.

        Returns:
            None
        """

        self.chosen_files = files
        logger.debug(f'Loaded files: {files}')

    def _load_default_option_class(self, ClassOfDefaultOptions):
        """
        Loads the default options class for the protocol window.

        Args:
            ClassOfDefaultOptions: The class representing the default options.

        Returns:
            None
        """
        self.ClassOfDefaultOptions = ClassOfDefaultOptions
        self.default_options = ClassOfDefaultOptions()
        logger.debug(
            f'Loaded default options: {self.default_options} of {ClassOfDefaultOptions}')

    def _handle_useDefault_event(self):
        """
        Handles the event when the 'Use Default' button is clicked.

        Args:
            None

        Returns:
            None
        """
        def on_click():
            self.load_default_operations()

        self.pushButton_useDefault.clicked.connect(on_click)

    def load_default_operations(self):
        """
        Loads default operations and handles option changes in the protocol window.
        ! It is designed to be executed **after** self.option_plainTextEdits is set.

        It does multiple things:
        1. Reload the chosen files;
        2. Handle option changes in the protocol window, it also checks the format of the options;
        3. It handles using default options operation to set back to the default options.

        Args:
            None

        Returns:
            None
        """

        # --------------------
        # 1. Reset the chosenFiles's list to chosen_files
        self.listWidget_chosenFiles.clear()
        n = len(self.chosen_files)
        self.listWidget_chosenFiles.addItems(
            [
                f"{i + 1} | {n}: {e['path'].as_posix()}"
                for i, e in enumerate(self.chosen_files)
            ]
        )
        logger.debug('Filled listWidget_chosenFiles')

        # --------------------
        # 2. Handle option changes in the protocol window
        default_options = self.ClassOfDefaultOptions()

        def option_changed():
            '''
            Convert the options into its corresponding type.
            If failed, raise warning and color the text into red color.
            '''
            for k, v in self.option_plainTextEdits.items():
                # Convert the options into its corresponding type
                # if failed, the option is None
                _type = type(default_options.__getattribute__(k))
                _type_name = str(_type).split('\'')[1]
                _text = v.toPlainText()

                option = None
                try:
                    cmd = f'{_type_name}({_text})'
                    option = eval(cmd)
                except Exception:
                    logger.warning(f'Failed convert to {_type}: {_text}')
                self.options[k] = option

                # Color the text with red color
                if option is None:
                    v.setStyleSheet('color:red;')
                else:
                    v.setStyleSheet('color:black;')

            # Clear existing analysis options,
            # and place them with the new values.
            self.listWidget_analysisOptions.clear()
            self.listWidget_analysisOptions.addItems(
                [f'{v}:\t{k}' for v, k in self.options.items()]
            )

            logger.debug(f'Changed options: {self.options}')

        # --------------------
        # 3. Reset the options with the default options
        for attr in [e for e in dir(default_options) if not e.startswith('_')]:
            value = default_options.__getattribute__(attr)
            try:
                self.option_plainTextEdits[attr].setPlainText(f'{value}')
                self.option_plainTextEdits[attr].textChanged.disconnect()
                self.option_plainTextEdits[attr].textChanged.connect(
                    option_changed)
                logger.debug(f'Set UI {attr} to {value}')
            except Exception as err:
                logger.warning(
                    f'Failed set UI {attr} to {value}, error: {err}')

        option_changed()

    def set_protocol_slogan(self, slogan: str = 'Unknown protocol'):
        self.label_protocolSlogan.setText(slogan)

# %% ---- 2024-05-09 ------------------------
# Play ground


# %% ---- 2024-05-09 ------------------------
# Pending


# %% ---- 2024-05-09 ------------------------
# Pending
