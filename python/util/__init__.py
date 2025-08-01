"""
File: __init__.py
Author: Chuncheng Zhang
Date: 2024-04-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Initialize the project's backend workers

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
from loguru import logger

from pathlib import Path
from .plotly_dash.thread_app import DashApp, html

# %% ---- 2024-04-23 ------------------------
# Function and class
project_name = 'EEG for everyone'

# __file__ -> util -> python -> project_root
project_root = Path(__file__).parent.parent.parent

logger_path = project_root.joinpath(f'log/{project_name}.log')
cache_path = project_root.joinpath('cache')
cache_path.mkdir(parents=True, exist_ok=True)
asset_path = project_root.joinpath('asset')


# %% ---- 2024-04-23 ------------------------
# Play ground

logger.add(logger_path, rotation='5 MB')


# %% ---- 2024-04-23 ------------------------
# Pending
# Initialize the dash_app
dash_app = DashApp()
# Handle the dynamic_html


@dash_app.app.server.route('/get_dynamic_html', methods=['GET'])
def _get_dynamic_html():
    return dash_app.dynamic_html


# %% ---- 2024-04-23 ------------------------
# Pending
