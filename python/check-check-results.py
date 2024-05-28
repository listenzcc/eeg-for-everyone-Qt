"""
File: check-check-results.py
Author: Chuncheng Zhang
Date: 2024-04-23
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


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
import pandas as pd

from util import logger, cache_path


# %% ---- 2024-04-23 ------------------------
# Function and class


# %% ---- 2024-04-23 ------------------------
# Play ground
if __name__ == '__main__':
    found_files = pd.read_pickle(cache_path.joinpath('found_files'))
    check_results = pd.read_pickle(cache_path.joinpath('check_results'))

    print(found_files)
    print(check_results)

    df = pd.merge(check_results, found_files, on='path')
    print(df)

    group = df.groupby(['status', 'protocol', 'format'])
    group = group[['path']]
    print(group.count())


# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
