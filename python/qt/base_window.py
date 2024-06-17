"""
File: base_window.py
Author: Chuncheng Zhang
Date: 2024-04-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The base class of UI window.
    It automatically loads the window and find its components.
    The components whose objectName starts with known_component_prefix are marked as the components.

    ! It raises the warnings when the UI window has something but the widget does not.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-25 ------------------------
# Requirements and constants
from PySide6 import QtCore, QtWidgets

from . import logger

# --------------------
known_component_prefix = 'zcc_'

# %% ---- 2024-04-25 ------------------------
# Function and class


class BaseWindow(object):
    """
    Initializes the BaseWindow object with a QtWidgets window and searches for children elements starting with 'zcc_'.

    Args:
        window: The QtWidgets window to initialize the BaseWindow object.

    Returns:
        dict: A dictionary containing children elements with keys as object names starting with 'zcc_'.

    Examples:
        base_window = BaseWindow(window)
        children = base_window._search_children()
    """

    window = None
    children = None
    known_component_prefix = known_component_prefix

    def __init__(self, window: QtWidgets):
        self.window = window
        self._search_children()
        self._assign_children()
        self._set_window_title()

    def _set_window_title(self, title: str = None):
        if title is None:
            title = '脑机接口专项'
        self.window.setWindowTitle(title)
        logger.debug(f'Set window title: {title}')

    def _search_children(self):
        """
        Searches for children elements within the window object that have object names starting with 'zcc_'.

        Returns:
            dict: A dictionary containing children elements with keys as object names starting with 'zcc_'.
        """

        raw = list(self.window.findChildren(
            QtCore.QObject,
            options=QtCore.Qt.FindChildrenRecursively
        ))
        children = {
            e.objectName(): e for e in raw
            if e.objectName().startswith(self.known_component_prefix)
        }
        logger.debug(f'Found children: {children}')
        self.children = children
        return children

    def _assign_children(self):
        for k, v in self.children.items():
            attr = k[len(self.known_component_prefix):]

            if not hasattr(self, attr):
                logger.warning(
                    f'Unknown attribute (UI has it, but window does not.): {attr}')

            self.__setattr__(attr, v)

            logger.debug(f'Assigned child: {attr} = {v}')

    def show(self):
        """
        Shows the window object.

        Returns:
            None
        """

        self.window.show()


# %% ---- 2024-04-25 ------------------------
# Play ground


# %% ---- 2024-04-25 ------------------------
# Pending


# %% ---- 2024-04-25 ------------------------
# Pending
