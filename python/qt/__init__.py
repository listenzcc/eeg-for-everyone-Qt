"""
File: __init__.py
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
import sys
from pathlib import Path

# ! --------------------
# ! Very important imports, it adds python folder to sys.path
p = Path(__file__).parent.parent  # noqa
sys.path.append(p.as_posix())  # noqa

from util import logger, dash_app, project_root, cache_path, asset_path

from util.default import default_options

from util.MI_analysis import MI_Analysis
from util.P300_analysis import P300_Analysis
from util.SSVEP_analysis import SSVEP_Analysis
from util.analysis.base_analysis import BaseAnalysis


# %% ---- 2024-04-25 ------------------------
# Function and class


# %% ---- 2024-04-25 ------------------------
# Play ground


# %% ---- 2024-04-25 ------------------------
# Pending


# %% ---- 2024-04-25 ------------------------
# Pending
