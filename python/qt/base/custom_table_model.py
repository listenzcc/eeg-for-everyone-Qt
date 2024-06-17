"""
File: custom_table_model.py
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

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QPersistentModelIndex


class CustomTableModel(QAbstractTableModel):
    def __init__(self):
        QAbstractTableModel.__init__(self)

    def load_dataFrame(self, df, display_columns: list = None):
        self.df = df.copy()

        if display_columns is None:
            self.display_df = df.copy()
        else:
            self.display_df = df[display_columns].copy()

        self.column_count = len(self.display_df.columns)
        self.row_count = len(self.display_df)

    def rowCount(self, parent=QModelIndex()):
        return self.row_count

    def columnCount(self, parent=QModelIndex()):
        return self.column_count

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.display_df.columns.tolist()[section]
        else:
            return f"{section}"

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            column = index.column()
            row = index.row()
            col = self.display_df.columns[column]
            value = self.display_df.iloc[row][col]
            return str(value)
        elif role == Qt.BackgroundRole:
            return QColor(Qt.white)
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignRight
        return None

    def bind_tableView(self, tableView):
        tableView.setModel(self)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        tableView.viewport().update()
        self.tableView = tableView

    def on_select(self, selected):
        selected_row = selected.toList()[0].top()
        self.tableView.viewport().update()
        return self.df.iloc[selected_row]


# %% ---- 2024-04-25 ------------------------
# Function and class

# %% ---- 2024-04-25 ------------------------
# Play ground


# %% ---- 2024-04-25 ------------------------
# Pending


# %% ---- 2024-04-25 ------------------------
# Pending
