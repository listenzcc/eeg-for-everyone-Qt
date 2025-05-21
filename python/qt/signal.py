"""
File: signal.py
Author: Chuncheng Zhang
Date: 2025-05-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Global signal and slot for the whole project.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-21 ------------------------
# Requirements and constants
from PySide6.QtCore import Signal, QThread


# %% ---- 2025-05-21 ------------------------
# Function and class
class GlobalSignal(QThread):
    """
    Global signal and slot for the whole project.
    """
    progress_bar_value = Signal(int)


GS = GlobalSignal()

# %% ---- 2025-05-21 ------------------------
# Play ground


# %% ---- 2025-05-21 ------------------------
# Pending


# %% ---- 2025-05-21 ------------------------
# Pending
