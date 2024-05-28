"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-02-21
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

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
from pathlib import Path
from omegaconf import OmegaConf

from data.search_data import RawDataFiles
from data.load_raw_data import LoadRawData


# %% ---- 2024-02-21 ------------------------
# Function and class


# %% ---- 2024-02-21 ------------------------
# Play ground
if __name__ == "__main__":
    backend_root = Path(__file__).parent
    conf = OmegaConf.load(backend_root.joinpath("conf/default.yaml"))
    print(conf)

    rdf = RawDataFiles(conf.data_folder)
    for e in rdf.find_all():
        lrd = LoadRawData(e)

# %% ---- 2024-02-21 ------------------------
# Pending


# %% ---- 2024-02-21 ------------------------
# Pending


# %%
