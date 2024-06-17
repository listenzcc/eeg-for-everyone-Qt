"""
File: find-files-with-format-check.py
Author: Chuncheng Zhang
Date: 2024-04-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Format check for the known data folders

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
import json
import time
import pandas as pd

from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from util import logger, cache_path
from util.find_files import find_files, format_check


# %% ---- 2024-04-23 ------------------------
# Function and class


# %% ---- 2024-04-23 ------------------------
# Play ground
if __name__ == '__main__':
    folders = [
        Path('D://脑机接口专项')
    ]

    tic = time.time()

    # --------------------
    dfs = []
    for folder in tqdm(folders, 'Searching for files'):
        # found_files = find_files(folder, limit=20)
        found_files = find_files(folder)
        dfs.append(found_files)
        logger.info(f'Checked folder: {folder}, for {len(found_files)} files')

    # --------------------
    found_files = pd.concat(dfs, axis=0)
    found_files.index = range(len(found_files))
    logger.info(f'Found files:\n{found_files}')

    # --------------------
    buffer = []
    for i, row in tqdm(found_files.iterrows(), 'Format checking'):
        output = format_check(row)
        buffer.append(output)
        # print(output)
    check_results = pd.DataFrame(buffer)

    # --------------------
    found_files.to_pickle(cache_path.joinpath('found_files'))
    check_results.to_pickle(cache_path.joinpath('check_results'))

    passed = time.time() - tic
    checks_info = dict(
        date=f'{datetime.now()}',
        cost=f'{passed:.2f} seconds'
    )
    json.dump(checks_info, open(
        cache_path.joinpath('latest_check_info.json'), 'w'))


# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
