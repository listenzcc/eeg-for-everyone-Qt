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
import matplotlib.pyplot as plt

from PIL import Image
from PIL.ImageQt import ImageQt

from PySide6 import QtWidgets
from PySide6.QtGui import QPixmap
from PySide6.QtUiTools import QUiLoader

from .base.base_window import BaseWindow
from . import logger, project_root, cache_path, asset_path


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
    buttonBox_submit = None

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

        logger.info(f'Initialized with obj: {analysis_obj}')

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

        self.on_select_file()

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
        for k in epochs.event_id:
            n = len(epochs[k])
            self.comboBox_selectEventId.addItem(f'{k} | {n}')

        def on_changed():
            self.handle_accept()
            self.comboBox_selectEventId.setFocus()

        self.comboBox_selectEventId.currentIndexChanged.disconnect()
        self.comboBox_selectEventId.currentIndexChanged.connect(on_changed)

        self.comboBox_selectFile.setFocus()

    def handle_accept(self):
        print('**** Analysis ****')
        # ----------------------------------------
        # ---- Get output's geometry ----
        width = self.label_output.geometry().width()
        height = self.label_output.geometry().height()
        logger.debug(f'The output size is {width}x{height}')

        # ----------------------------------------
        # ---- Place the computing img ----
        pixmap = img_to_pixmap(asset_imgs.get('computing'), width, height)
        self.label_output.setPixmap(pixmap)
        self.label_output.repaint()

        # ----------------------------------------
        # ---- Check current options ----
        idx = self.comboBox_selectFile.currentIndex()
        event_id = self.comboBox_selectEventId.currentText()
        event_id = event_id.split('|')[0].strip()
        method_name = self.comboBox_selectMethod.currentText()
        logger.debug(f'Submit for {method_name}, {idx}, {event_id}')

        if not len(event_id):
            logger.warning(f'Invalid event_id: {event_id}')
            return

        # ----------------------------------------
        # ---- Compute with necessary options: idx and event_id ----
        self._toggle_input_components(False)
        try:
            fig = self.analysis_obj.methods[method_name](
                idx, event_id)
            fig.canvas.draw()
            # Clear existing plt buffer
            plt.clf()

            # ----------------------------------------
            # ---- Fit image to output frame ----
            img = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb())  # .resize((width, height))

            pixmap = img_to_pixmap(img, width, height)
            self.label_output.setPixmap(pixmap)

        except Exception as err:
            import traceback
            pixmap = img_to_pixmap(asset_imgs.get('error'), width, height)
            self.label_output.setPixmap(pixmap)

            # ----------------------------------------
            # ---- Popup error box ----
            dialog = QtWidgets.QDialog(parent=self.window)
            error_title = f'Computing on method: {
                method_name} got error: {err}'
            dialog.setWindowTitle(error_title)
            layout = QtWidgets.QVBoxLayout()
            message = QtWidgets.QLabel(traceback.format_exc())
            layout.addWidget(message)
            dialog.setLayout(layout)
            dialog.show()

            logger.error(error_title)

        finally:
            self._toggle_input_components(True)
            return

    def _toggle_input_components(self, enabled: bool = True):
        self.comboBox_selectFile.setEnabled(enabled)
        self.comboBox_selectEventId.setEnabled(enabled)
        self.comboBox_selectMethod.setEnabled(enabled)
        self.buttonBox_submit.setEnabled(enabled)
        logger.debug(f'Toggle input components as enabled: {enabled}')

# %% ---- 2024-06-14 ------------------------
# Play ground


# %% ---- 2024-06-14 ------------------------
# Pending


# %% ---- 2024-06-14 ------------------------
# Pending
