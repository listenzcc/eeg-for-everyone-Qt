"""
File: window_of_data_selection.py
Author: Chuncheng Zhang
Date: 2024-04-25
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


# %% ---- 2024-04-25 ------------------------
# Requirements and constants
import pandas as pd
from PySide6.QtUiTools import QUiLoader

from .base.base_window import BaseWindow
from .base.custom_table_model import CustomTableModel
# from .window_of_MI import MIWindow
from .window_of_setup_options import SetupOptionsWindow
from . import logger, project_root, cache_path

# --------------------
loader = QUiLoader()

# %% ---- 2024-04-25 ------------------------
# Function and class


def pretty_traceback(message):
    return '\n'.join(message) if isinstance(message, list) else message


def pretty_dict(dict_like, leading='\n  |\t'):
    if isinstance(dict_like, dict):
        return leading+leading.join(f'{k}:\t{v}' if k.lower() != 'traceback' else f'{k}:\t{pretty_traceback(v)}' for k, v in dict_like.items())
    else:
        return f'{dict_like}'


def load_files_table(map):
    found_files = pd.read_pickle(cache_path.joinpath('found_files'))
    check_results = pd.read_pickle(cache_path.joinpath('check_results'))

    df = pd.merge(check_results, found_files, on='path')
    df = df[df.apply(map, axis=1)]

    logger.debug(f'Loaded files: {len(df)}')

    return df


class DataSelectionWindow(BaseWindow):
    layout_path = project_root.joinpath('layout/data_selection.ui')

    # --------------------
    # Known components
    # --
    comboBox_protocolSelection = None
    # --
    lineEdit_pathFilter = None
    label_filteredFiles = None
    tableView_validDataSelection = None
    pushButton_chooseAllFiles = None
    textBrowser_candidateFileDetail = None
    pushButton_chooseFile = None
    pushButton_clearChosenFiles = None
    listWidget_chosenFiles = None
    # --
    listWidget_failedFiles = None
    textBrowser_failedFiles = None
    # --
    tableView_protocolSummary = None
    # --
    buttonBox_goToNext = None

    # --------------------
    # The table of found files
    files_table = load_files_table(lambda se: se['status'] != 'failed')
    failed_files_table = load_files_table(lambda se: se['status'] == 'failed')

    # --------------------
    # variables
    protocol = None
    candidate_file = None
    chosen_files = []
    filtered_files = []

    def __init__(self):
        super().__init__(loader.load(self.layout_path, None))
        self.connect_choose_file_functions()
        self.setup_protocols()
        self.update_protocolSummary()
        self.handle_goToNext_events()

    def handle_goToNext_events(self):

        def _accept():
            unknown_protocol = True
            unknown_protocol = False

            window = SetupOptionsWindow(
                self.chosen_files, self.protocol, self.window)
            window.show()

            if unknown_protocol:
                logger.warning('Not support protocol: {self.protocol}')

        self.buttonBox_goToNext.accepted.connect(_accept)

    def update_protocolSummary(self):
        df = pd.concat([self.files_table, self.failed_files_table], axis=0)
        group = df.groupby(['status', 'protocol', 'format'])
        group = group[['path']]
        count = group.count()

        # Convert multiple index to columns
        count = count.reset_index()

        # Put the data frame of the count into the tableView_protocolSummary
        model = CustomTableModel()
        model.load_dataFrame(count)
        model.bind_tableView(self.tableView_protocolSummary)

    def setup_protocols(self):
        """
        Sets up the protocols for selection and connects the protocol selection event.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        protocols = sorted(set(self.files_table['protocol'].to_list()))
        logger.debug(f'Found protocols: {protocols}')

        self.comboBox_protocolSelection.addItems(protocols)

        def _select_protocol(protocol: str):
            self.protocol = protocol
            self.refresh_with_current_protocol()
            self.refresh_failed_files_with_current_protocol()
            logger.debug(f'Selected protocol: {protocol}')

        self.comboBox_protocolSelection.currentTextChanged.connect(
            _select_protocol)
        _select_protocol(self.comboBox_protocolSelection.currentText())

    def connect_choose_file_functions(self):
        """
        Connects the choose file and discard clicked file functions to their respective UI elements.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # --------------------
        # Launch choose file operation and related functions

        def _choose_candidate_file():
            if self.candidate_file is None:
                return
            if self.candidate_file not in self.chosen_files:
                self.chosen_files.append(self.candidate_file)
            self.update_listWidget_chosenFiles()

        self.pushButton_chooseFile.clicked.connect(_choose_candidate_file)

        def _choose_all_filtered_files():
            for file in self.filtered_files:
                if file not in self.chosen_files:
                    self.chosen_files.append(file)
            self.update_listWidget_chosenFiles()

        self.pushButton_chooseAllFiles.clicked.connect(
            _choose_all_filtered_files)

        def _discard_clicked_file():
            idx = self.listWidget_chosenFiles.currentRow()
            file = self.chosen_files.pop(idx)
            self.update_candidate_file(file)
            logger.debug(f'Discard file: {file}')
            self.update_listWidget_chosenFiles()

        self.listWidget_chosenFiles.itemDoubleClicked.connect(
            _discard_clicked_file)

        def _only_update_candidate_file():
            idx = self.listWidget_chosenFiles.currentRow()
            file = self.chosen_files[idx]
            self.update_candidate_file(file)

        self.listWidget_chosenFiles.itemClicked.connect(
            _only_update_candidate_file)

        def _clear_all_chosen_files():
            self.chosen_files = []
            self.update_listWidget_chosenFiles()
            logger.debug('Clear all the chosen files')

        self.pushButton_clearChosenFiles.clicked.connect(
            _clear_all_chosen_files)

    def refresh_failed_files_with_current_protocol(self):
        df = self.failed_files_table.copy()
        df = df[df['protocol'] == self.protocol]
        files = sorted(
            [e for i, e in df.iterrows()],
            key=lambda se: se['path'])

        logger.debug(f'Found failed files: {files}')

        def _select_failed_file():
            idx = self.listWidget_failedFiles.currentRow()
            file = files[idx]
            text = '\n\n'.join(
                f'{k}: \t{pretty_dict(v)}'
                for k, v in file.items())
            self.textBrowser_failedFiles.setText(text)

        self.listWidget_failedFiles.clear()
        self.listWidget_failedFiles.addItems(
            [f"{i}: {e['path']}" for i, e in enumerate(files)])
        self.listWidget_failedFiles.clicked.disconnect()
        self.listWidget_failedFiles.clicked.connect(_select_failed_file)
        self.textBrowser_failedFiles.setText('')

    def refresh_with_current_protocol(self):
        """
        Refreshes the data based on the current protocol selected, including clearing variables and updating the displayed data.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # --------------------
        # Clear necessary variables
        self.candidate_file = None
        self.chosen_files = []
        self.update_listWidget_chosenFiles()
        self.update_candidate_file()

        # --------------------
        model = CustomTableModel()
        display_columns = ['protocol', 'path', 'status', 'format']

        # --------------------
        # Path filter

        def _path_filter_text_changed(text):
            logger.debug(f'Changed pathFilter text: {text}')

            df = self.files_table.copy()
            df = df[df['protocol'] == self.protocol]
            n = len(df)
            df = df[df['path'].map(
                lambda e: text.upper() in str(e).upper())]
            m = len(df)
            self.label_filteredFiles.setText(f'{m: 4d} | {n: 4d}')
            self.filtered_files = [dict(se) for i, se in df.iterrows()]

            model.load_dataFrame(df, display_columns=display_columns)
            model.bind_tableView(self.tableView_validDataSelection)
            model.tableView.selectionModel().selectionChanged.disconnect()
            model.tableView.selectionModel().selectionChanged.connect(_on_select_file)
            model.tableView.doubleClicked.disconnect()
            model.tableView.doubleClicked.connect(_on_double_clicked)
            logger.debug('Updated tableView')

        self.lineEdit_pathFilter.setText('')
        self.lineEdit_pathFilter.textChanged.disconnect()
        self.lineEdit_pathFilter.textChanged.connect(_path_filter_text_changed)

        # --------------------
        # Files table
        def _on_select_file(selected, deselected):
            # The selected and deselected are both like
            # [<PySide6.QtCore.QItemSelectionRange(QModelIndex(68,0,0x0,CustomTableModel(0x25b5ed2b0a0)),QModelIndex(68,3,0x0,CustomTableModel(0x25b5ed2b0a0))) at 0x0000025B6150C300>]

            se = model.on_select(selected)
            file = dict(se)
            self.update_candidate_file(file)

        def _on_double_clicked(pnt):
            file = self.filtered_files[pnt.row()]
            self.update_candidate_file(file)

            if file not in self.chosen_files:
                self.chosen_files.append(file)
            self.update_listWidget_chosenFiles()

        df = self.files_table.copy()
        df = df[df['protocol'] == self.protocol]
        n = len(df)
        self.label_filteredFiles.setText(f'{n: 4d} | {n: 4d}')
        self.filtered_files = [dict(se) for i, se in df.iterrows()]

        model.load_dataFrame(df, display_columns=display_columns)
        model.bind_tableView(self.tableView_validDataSelection)
        model.tableView.selectionModel().selectionChanged.disconnect()
        model.tableView.selectionModel().selectionChanged.connect(_on_select_file)
        model.tableView.doubleClicked.disconnect()
        model.tableView.doubleClicked.connect(_on_double_clicked)
        self.tableView_validDataSelection.setColumnWidth(1, 400)

    def update_candidate_file(self, file: dict = None):
        """
        Updates the candidate file details in the text browser.

        Args:
            file (dict): The candidate file to update.

        Returns:
            None
        """

        self.candidate_file = file

        text = ''
        if file is not None:
            text = '\n\n'.join(
                f'{k}:\t{pretty_dict(v)}'
                for k, v in file.items())

        self.textBrowser_candidateFileDetail.setText(text)

        logger.debug(f'Changed candidate file: {file}')

    def update_listWidget_chosenFiles(self):
        """
        Updates the list widget with the chosen files.

        ! It makes sure the same size and order between the chosen_files and listWidget_chosenFiles

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        self.chosen_files.sort(key=lambda x: x['path'])
        self.listWidget_chosenFiles.clear()
        self.listWidget_chosenFiles.addItems(
            [
                f"{i + 1}: {e['path'].as_posix()}"
                for i, e in enumerate(self.chosen_files)
            ]
        )

# %% ---- 2024-04-25 ------------------------
# Play ground


# %% ---- 2024-04-25 ------------------------
# Pending


# %% ---- 2024-04-25 ------------------------
# Pending
