"""
File: window_of_analysis_result.py
Author: Chuncheng Zhang
Date: 2024-06-14
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


# %% ---- 2024-06-14 ------------------------
# Requirements and constants
import time
import random
import traceback
import contextlib
import webbrowser
import matplotlib.pyplot as plt

from pathlib import Path
from threading import Thread
from datetime import datetime

from PIL import Image
from PIL.ImageQt import ImageQt

from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6.QtGui import QPixmap
from PySide6.QtUiTools import QUiLoader

from .base.base_window import BaseWindow
from .report.pdf import PDFReportGenerator
from . import logger, dash_app, project_root, cache_path, asset_path


# ----------------------------------------
# ---- Constants ----
loader = QUiLoader()
layout_path = project_root.joinpath('layout/analysis_results.ui')

asset_imgs = dict(
    computing=Image.open(asset_path.joinpath('img/computing.png')),
    error=Image.open(asset_path.joinpath('img/error.png'))
)
computing_img = Image.open(asset_path.joinpath('img/computing.png'))

# %% ---- 2024-06-14 ------------------------
# Function and class


def img_to_pixmap(img, width, height):
    r = img.width / img.height
    if r > width/height:
        # if img is wider
        img = img.resize((width, int(width/r)))
    else:
        # if img is taller
        img = img.resize((int(height*r), height))

    return QPixmap.fromImage(ImageQt(img))


class AnalysisResultsWindow(BaseWindow):
    # ----------------------------------------
    # ---- Known components ----
    label_output: QtWidgets.QLabel = None
    comboBox_selectFile: QtWidgets.QComboBox = None
    comboBox_selectEventId: QtWidgets.QComboBox = None
    comboBox_selectMethod: QtWidgets.QComboBox = None
    label_progressing: QtWidgets.QLabel = None
    radioButton_requireDetail: QtWidgets.QRadioButton = None
    pushButton_fetchData: QtWidgets.QPushButton = None
    pushButton_viewInWeb: QtWidgets.QPushButton = None
    pushButton_generateReport: QtWidgets.QPushButton = None
    buttonBox_submit = None
    timer = QtCore.QTimer()

    # ----------------------------------------
    # ---- Input ----
    analysis_obj = None

    def __init__(self, analysis_obj, parent):
        window = loader.load(layout_path, parent)
        super(AnalysisResultsWindow, self).__init__(window)
        self._set_window_title(f'Results of {analysis_obj.protocol}')

        self.on_load(analysis_obj)

        self.load_methods()

        self.buttonBox_submit.accepted.connect(self.handle_accept)

        self.handle_accept()
        # self._timer()

        logger.info(f'Initialized with obj: {analysis_obj}')

    def _timer(self):
        def _timeout():
            self.label_progressing.repaint()

        self.timer.timeout.connect(_timeout)

        self.timer.start(100)

    def _get_selected_options(self):
        '''
        Get the options from the user operating the UI

        - selected_file_idx: The selected file idx;
        - selected_event_id: The event id.
        '''
        selected_file_idx = self.comboBox_selectFile.currentIndex()
        selected_event_id = self.comboBox_selectEventId.currentText()
        selected_event_id = selected_event_id.split(':')[0].strip()
        return selected_file_idx, selected_event_id

    def handle_pushButton_fetchData(self):
        def _handle():
            fileName = QtWidgets.QFileDialog.getSaveFileName(
                caption='File name to save', filter='Matlab Compatible Files (*.mat)')

            # Do nothing when not selecting any file
            if len(fileName[0]) == 0:
                return

            path = Path(fileName[0])
            selected_file_idx, selected_event_id = self._get_selected_options()
            self.analysis_obj._save_data(
                selected_file_idx, selected_event_id, path=path)

        self.pushButton_fetchData.clicked.connect(_handle)

    def handle_pushButton_viewInWeb(self):
        def _handle():
            webbrowser.open('http://localhost:8890')
            pass

        self.pushButton_viewInWeb.clicked.connect(_handle)

    def handle_pushButton_generateReport(self):
        def _handle():
            print('Clicked generateReport button.')
            self.generate_report()
            pass

        self.pushButton_generateReport.clicked.connect(_handle)

    def generate_report(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(
            caption='Report file name', filter='PDF Files (*.pdf)')

        # Do nothing when not selecting any file
        if len(fileName[0]) == 0:
            return

        path = Path(fileName[0])
        print(path)
        pass

    def on_load(self, analysis_obj):
        # ----------------------------------------
        # ---- Load analysis object ----
        self.analysis_obj = analysis_obj

        # ----------------------------------------
        # ---- Initialize file options ----
        self.comboBox_selectFile.clear()
        for file in analysis_obj.files:
            self.comboBox_selectFile.addItem(file['short_name'])

        self.comboBox_selectFile.currentIndexChanged.connect(
            lambda e: self.on_select_file())

        # Handle select file operation.
        self.on_select_file()

        # Handle save data operation.
        self.handle_pushButton_fetchData()

        # Handle view in web operation.
        self.handle_pushButton_viewInWeb()

        # Handle generate report operation.
        self.handle_pushButton_generateReport()

        return analysis_obj

    def load_methods(self):
        # ----------------------------------------
        # ---- Load methods ----
        self.comboBox_selectMethod.clear()
        for method in self.analysis_obj.methods:
            self.comboBox_selectMethod.addItem(method)
            logger.debug(f'Appended method: {method}')

        def on_changed():
            self.handle_accept()
            self.comboBox_selectMethod.setFocus()

        self.comboBox_selectMethod.currentIndexChanged.disconnect()
        self.comboBox_selectMethod.currentIndexChanged.connect(on_changed)

    def on_select_file(self, idx: int = None):
        '''
        The file idx is selected,
        update the self.comboBox_selectEventId by the evoked_id of the epochs 
        '''
        if idx is None:
            idx = self.comboBox_selectFile.currentIndex()

        epochs = self.analysis_obj.objs[idx].epochs

        self.comboBox_selectEventId.clear()
        logger.debug(f'Selected epochs: {epochs}')
        for k, v in epochs.event_id.items():
            n = len(epochs[k])
            self.comboBox_selectEventId.addItem(f'{k}\t: {v}\t({n} items)')

        def on_changed():
            self.handle_accept()
            self.comboBox_selectEventId.setFocus()

        self.comboBox_selectEventId.currentIndexChanged.disconnect()
        self.comboBox_selectEventId.currentIndexChanged.connect(on_changed)

        self.comboBox_selectFile.setFocus()

    def handle_accept(self):
        '''
        Handle the accept event for file selection, event selection and method selection.
        '''

        # ----------------------------------------
        # ---- Check current options ----
        selected_file_idx, selected_event_id = self._get_selected_options()
        flag_require_detail = self.radioButton_requireDetail.isChecked()
        method_name = self.comboBox_selectMethod.currentText()

        lst = [
            f'AnalysisObj: {type(self.analysis_obj)}',
            f'Method: {method_name}',
            f'fileIdx: {selected_file_idx}',
            f'eventId: {selected_event_id}'
        ]
        print('\n\n********************************************************************************')
        [print(f'**** {e} ****') for e in lst]
        logger.debug(', '.join(lst))

        # ----------------------------------------
        # ---- Compute with necessary options: idx and event_id ----
        if not len(selected_event_id):
            logger.warning(f'Invalid event_id: {selected_event_id}')
            return

        # ----------------------------------------
        # ---- Get output's geometry ----
        width = self.label_output.geometry().width()
        height = self.label_output.geometry().height()
        logger.debug(f'The output geometry size is {width}x{height}')

        # ----------------------------------------
        # ---- Place the computing img ----
        with contextlib.suppress(Exception):
            pixmap = img_to_pixmap(asset_imgs.get('computing'), width, height)
            self.label_output.setPixmap(pixmap)

        # Clear the dash_app's children part
        dash_app.div.children = []

        # Protect the input components from repeated calls.
        self._toggle_input_components(False)
        try:
            # Start progressing bar updating
            Thread(target=self._progress_bar_engage, daemon=True).start()

            # ! Call the method.
            fig = self.analysis_obj.methods[method_name](
                selected_file_idx,
                selected_event_id,
                flag_require_detail=flag_require_detail)
            fig.canvas.draw()

            # ----------------------------------------
            # ---- Fit image to output frame ----
            img = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb())  # .resize((width, height))

            # Clear existing plt buffer
            plt.close(fig)

            pixmap = img_to_pixmap(img, width, height)
            self.label_output.setPixmap(pixmap)

        except Exception as err:
            # Stop the progressing bar
            self._progress_bar_going_flag = False

            # Report traceback msg to console
            msg = traceback.format_exc()
            print(msg)

            # Draw the error image
            with contextlib.suppress(Exception):
                pixmap = img_to_pixmap(asset_imgs.get('error'), width, height)
                self.label_output.setPixmap(pixmap)

            # ----------------------------------------
            # ---- Popup error box ----
            dialog = QtWidgets.QDialog(parent=self.window)
            error_title = f'Failed: {method_name}. Error: {err}'
            dialog.setWindowTitle(error_title)
            layout = QtWidgets.QVBoxLayout()
            message = QtWidgets.QLabel(msg)
            layout.addWidget(message)
            dialog.setLayout(layout)
            dialog.exec()

            logger.error(error_title)

            dash_app.div.children.append(error_title)
            dash_app.div.children.append(msg)

        finally:
            # Release locking input components and finish the progress bar updating
            self._toggle_input_components(True)
            self._progress_bar_going_flag = False
            return

    def _toggle_input_components(self, enabled: bool = True):
        self.comboBox_selectFile.setEnabled(enabled)
        self.comboBox_selectEventId.setEnabled(enabled)
        self.comboBox_selectMethod.setEnabled(enabled)
        self.buttonBox_submit.setEnabled(enabled)
        logger.debug(f'Toggle input components as enabled: {enabled}')

    def _progress_bar_engage(self):
        self._progress_bar_going_flag = True

        tic = time.time()

        while self._progress_bar_going_flag:
            time.sleep(random.random() * 0.1)
            txt = pseudo_progressing_report()
            passed = time.time() - tic
            self.label_progressing.setText(f'{passed:0.2f} | {txt}')

            # ! The repaint method is dangerous in the thread
            # ! But it keeps updating during the backend computing
            # self.label_progressing.repaint()

        self.label_progressing.setText(
            f'Cost {passed:0.2f} seconds | Finished at {datetime.now()}')


def pseudo_progressing_report():
    chars = 'abcdefghijklmnopqrstuvwxyz           '
    k = random.randint(10, 30)
    return f'Computing ...'
    return ''.join(random.choices(chars, k=k))


# %% ---- 2024-06-14 ------------------------
# Play ground


# %% ---- 2024-06-14 ------------------------
# Pending


# %% ---- 2024-06-14 ------------------------
# Pending
