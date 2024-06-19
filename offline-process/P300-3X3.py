"""
File: P300-3X3.py
Author: Chuncheng Zhang
Date: 2024-06-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Append label.csv to each data in the P300(3X3) files

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-18 ------------------------
# Requirements and constants
import mne
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

# %%
# Load P300(3X3) data
cache_folder = Path(__file__).parent.parent.joinpath('cache')

found_files = pd.read_pickle(cache_folder.joinpath('found_files'))
check_results = pd.read_pickle(cache_folder.joinpath('check_results'))

data = pd.merge(check_results, found_files, on='path')
data = data[data['status'] != 'failed']
data = data[data['protocol'] == 'P300(3X3)']
data

# %% ---- 2024-06-18 ------------------------
# Function and class


def fetch_events(selected):
    """
    Fetches events from the selected data file and its corresponding event file.

    Args:
        selected: Dictionary containing 'path' for data file and 'evt_path' for event file.

    Returns:
        Tuple containing events and event IDs.
    """

    raw = mne.io.read_raw(selected['path'])
    annotations = mne.read_annotations(selected['evt_path'])
    raw.set_annotations(annotations)

    events, event_id = mne.events_from_annotations(raw)
    event_id_inv = {v: k for k, v in event_id.items()}
    return events, event_id, event_id_inv

# %% ---- 2024-06-18 ------------------------
# Play ground


for _, selected in tqdm(data.iterrows()):
    events, event_id, event_id_inv = fetch_events(selected)
    real_events = [event_id_inv[e] for e in events[:, -1]]

    event_ids_mark_as_target = []

    label = []
    for eid in real_events:
        # Mark as target:2 if is target event_id;
        # Mark as non-target:1 otherwise.
        label.append(2 if eid in event_ids_mark_as_target else 1)

        # Setup new target
        # 101->1, 102->2, ..., 109->9
        i = int(eid)
        if i in range(101, 109):
            event_ids_mark_as_target = [f'{i-100}']

    # Make and save dataFrame
    df = pd.DataFrame(events, columns=['timestamp', 'duration', 'event'])
    df['label'] = label
    df['_real_event'] = real_events
    dst = selected['path'].parent.joinpath('label.csv')
    df.to_csv(dst)

# %% ---- 2024-06-18 ------------------------
# Pending

# %% ---- 2024-06-18 ------------------------
# Pending
