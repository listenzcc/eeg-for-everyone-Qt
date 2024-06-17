"""
File: app.py
Author: Chuncheng Zhang
Date: 2024-04-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The application entrance for the project.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
import sys
from PySide6 import QtWidgets
from qt.window_of_data_selection import DataSelectionWindow


# %% ---- 2024-04-23 ------------------------
# Function and class


# %% ---- 2024-04-23 ------------------------
# Play ground
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    dsw = DataSelectionWindow()
    dsw.show()

    app.exec()

# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
