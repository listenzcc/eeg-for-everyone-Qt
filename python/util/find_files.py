"""
File: find_files.py
Author: Chuncheng Zhang
Date: 2024-04-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Find all the legal files in the given directory

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-23 ------------------------
# Requirements and constants
import os
import mne
import json
import traceback
import numpy as np
import pandas as pd

from pathlib import Path

from typing import Any
from tqdm.auto import tqdm

from . import logger, project_root
from .load_raw import RawObject


# %% ---- 2024-04-23 ------------------------
# Function and class
class AttrDict(dict):
    def __setattr__(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        try:
            # Get the supposed-to-be attribute
            return super().__getattribute__(name)
        except AttributeError as e:
            # If not exists, search for key-value pairs
            if name not in self:
                raise AttributeError(
                    f'Attribute "{name}" does not exist') from e
            return super().get(name)


class EEG_File(AttrDict):
    protocol = None
    path = None
    evt_path = None
    format = None
    short_name = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v


def parse_as_eeg_file_path(path: Path, folder: Path):
    output = None

    protocol = guess_protocol(path, folder)

    if protocol is None:
        return output

    name = path.name
    short_name = path.relative_to(folder).as_posix()

    if name.endswith('.cnt'):
        output = EEG_File(
            path=path, short_name=short_name, protocol=protocol, format='.cnt')

    if name == 'data.bdf':
        evt_path = path.parent.joinpath('evt.bdf')

        output = EEG_File(
            path=path, short_name=short_name, protocol=protocol, format='.bdf')

        if evt_path.is_file():
            output = EEG_File(
                path=path, evt_path=evt_path, short_name=short_name, protocol=protocol, format='.bdf')

    return output


def read_known_protocols():
    dct = json.load(open(project_root.joinpath(
        'asset/protocols.json'), encoding='utf8'))
    logger.debug(f'Loaded known protocols: {dct}')
    return dct


known_protocols = read_known_protocols()


def guess_protocol(path: Path, folder: Path) -> str:
    relative = path.relative_to(folder)
    name = relative.parts[0]
    return name if name in known_protocols else None


def find_files(folder: Path, limit: int = 1e6) -> list:
    buffer = []

    # Each element of the res is (folder, subfolder, files)
    res = os.walk(folder.as_posix())
    for a, b, c in tqdm(res, 'Searching for files'):
        for d in c:
            path = Path(a, d)
            efile = parse_as_eeg_file_path(path, folder)
            if efile is not None:
                buffer.append(efile)

        if len(buffer) > limit:
            break

    return pd.DataFrame(buffer)


def format_check(file: pd.Series):
    """
    Performs format checks on the provided file and returns the status and checks results.

    Args:
        file (pd.Series): The file to perform format checks on.

    Returns:
        dict: A dictionary containing the path of the file and its status along with any checks performed.
    """

    # obj = RawObject(file)
    output = dict(path=file['path'])

    obj, suspects = _check_basic(file)

    # Can not load
    if obj is None:
        output |= dict(status='failed', suspects=suspects)
        return output

    # ----------------------------------------
    # ---- SSVEP check ----
    if file['protocol'] == 'SSVEP':
        suspects, checks = _check_SSVEP(obj.raw)

        if any(suspects.values()):
            output |= dict(status='failed', checks=checks, suspects=suspects)
            logger.warning(f'SSVEP checks failed: {output}')
        else:
            output |= dict(status='passed', checks=checks, suspects=suspects)

        return output

    # ----------------------------------------
    # ---- P300 check ----
    if file['protocol'].startswith('P300'):
        suspects, checks = _check_P300(obj.raw)

        if any(suspects.values()):
            output |= dict(status='failed', checks=checks, suspects=suspects)
            logger.warning(f'P300 checks failed: {output}')
        else:
            output |= dict(status='passed', checks=checks, suspects=suspects)

        return output

    # ----------------------------------------
    # ---- MI check ----
    if file['protocol'].startswith('MI'):
        suspects, checks = _check_MI(obj.raw)

        if any(suspects.values()):
            output |= dict(status='failed', checks=checks, suspects=suspects)
            logger.warning(f'MI checks failed: {output}')
        else:
            output |= dict(status='passed', checks=checks, suspects=suspects)

        return output

    # Loaded but not checked
    # ! I think autoPassed is a better name
    output |= dict(status='autoPassed', checks=[], suspects=[])
    return output


def _check_basic(file):
    suspects = {}
    obj = None

    try:
        obj = RawObject(file)
    except Exception:
        suspects['traceback'] = [traceback.format_exc()]

    return obj, suspects


def _check_P300(raw):
    # Something I want to know
    checks = dict(
        ch_names=[],
        sfreq=None,
        event_id={},
        total_length=None
    )

    # Something is wrong
    suspects = dict(
        channels=[],
        sfreq=[],
        n_events=[],
    )

    def _check():
        # --------------------
        # Fetch information
        ch_names = raw.info['ch_names']
        sfreq = raw.info['sfreq']
        events, event_id = mne.events_from_annotations(raw)
        total_length = np.max([e[0] for e in events]) / \
            sfreq if len(events) > 0 else None
        checks.update(
            ch_names=ch_names,
            sfreq=sfreq,
            event_id=event_id,
            total_length=total_length)

        # --------------------
        # Check channels
        must_ch_names = [
            e.strip().upper()
            for e in 'Fz,F3,F4,Cz,C3,C4,CP1,CP2,CP5,CP6,Pz,P3,P4,P7,P8,POz,PO3,PO4,PO7,PO8,Oz,O1,O2'.split(',') if e.strip()]
        ch_names = [e.upper() for e in ch_names]
        for e in [e for e in must_ch_names if e not in ch_names]:
            suspects['channels'].append(
                f'The ch_names must contain {e}, which does not'
            )

        # --------------------
        # Check sfreq not be less than 250 Hz
        if sfreq < 250:
            suspects['sfreq'].append(
                f'The sfreq must not be less than 250 Hz, but the value is {
                    sfreq}'
            )

        # --------------------
        # Filter the valid events
        # Check events minimum number not less than 9 kinds
        event_id_inv = {v: k for k, v in event_id.items()}
        events = [e for e in events if int(
            event_id_inv[e[2]]) > 0 and int(event_id_inv[e[2]]) < 100]
        n_events = {e[2] for e in events}
        if len(n_events) < 9:
            suspects['n_events'].append(
                f'The (1-99) events should be equal or larger than 9 kinds, but the value is {
                    len(n_events)}'
            )

    try:
        _check()
    except Exception:
        suspects['traceback'] = [traceback.format_exc()]

    return suspects, checks


def _check_MI(raw):
    # Something I want to know
    checks = dict(
        ch_names=[],
        sfreq=None,
        event_id={},
        total_length=None
    )

    # Something is wrong
    suspects = dict(
        channels=[],
        sfreq=[],
        n_events=[],
        total_length=[],
    )

    def _check():
        # --------------------
        # Fetch information
        ch_names = raw.info['ch_names']
        sfreq = raw.info['sfreq']
        events, event_id = mne.events_from_annotations(raw)
        total_length = np.max([e[0] for e in events]) / \
            sfreq if len(events) > 0 else None

        checks.update(
            ch_names=ch_names,
            sfreq=sfreq,
            event_id=event_id,
            total_length=total_length)

        # --------------------
        # Check channels
        must_ch_names = [
            e.strip().upper()
            for e in 'C3, C4, Cz'.split(',') if e.strip()]
        ch_names = [e.upper() for e in ch_names]

        for e in [e for e in must_ch_names if e not in ch_names]:
            suspects['channels'].append(
                f'The ch_names must contain {e}, which does not'
            )

        # --------------------
        # Check sfreq not be less than 250 Hz
        if sfreq < 250:
            suspects['sfreq'].append(
                f'The sfreq must not be less than 250 Hz, but the value is {
                    sfreq}'
            )

        # --------------------
        # Check total length not be less than 180 seconds
        if total_length is None or total_length < 180:
            suspects['total_length'].append(
                f'The total length must not be less than 180 seconds, but the value is {
                    total_length}'
            )

    try:
        _check()
    except Exception:
        suspects['traceback'] = [traceback.format_exc()]

    # --------------------
    # No message is good message
    return suspects, checks


def _check_SSVEP(raw):
    # Something I want to know
    checks = dict(
        ch_names=[],
        sfreq=None,
        event_id={},
        total_length=None
    )

    # Something is wrong
    suspects = dict(
        channels=[],
        sfreq=[],
        n_events=[],
        total_length=[],
        min_gap=[],
    )

    def _check():
        # --------------------
        # Fetch information
        ch_names = raw.info['ch_names']
        sfreq = raw.info['sfreq']
        events, event_id = mne.events_from_annotations(raw)
        total_length = np.max([e[0] for e in events]) / \
            sfreq if len(events) > 0 else None

        checks.update(
            ch_names=ch_names,
            sfreq=sfreq,
            event_id=event_id,
            total_length=total_length)

        # --------------------
        # Check channels
        must_ch_names = [
            e.strip().upper()
            for e in 'PO3,PO5,POz,PO4,PO6,O1,Oz,O2'.split(',') if e.strip()]
        ch_names = [e.upper() for e in ch_names]

        for e in [e for e in must_ch_names if e not in ch_names]:
            suspects['channels'].append(
                f'The ch_names must contain {e}, which does not'
            )

        # --------------------
        # Check sfreq not be less than 250 Hz
        if sfreq < 250:
            suspects['sfreq'].append(
                f'The sfreq must not be less than 250 Hz, but the value is {
                    sfreq}'
            )

        # --------------------
        # Check total length not be less than 180 seconds
        if total_length is None or total_length < 180:
            suspects['total_length'].append(
                f'The total length must not be less than 180 seconds, but the value is {
                    total_length}'
            )

        # --------------------
        # Filter the valid events
        # Check events minimum number not less than 10 kinds
        event_id_inv = {v: k for k, v in event_id.items()}
        events = [e for e in events if int(
            event_id_inv[e[2]]) > 0 and int(event_id_inv[e[2]]) < 241]
        n_events = {e[2] for e in events}
        if len(n_events) < 10:
            suspects['n_events'].append(
                f'The (1-240) events should be equal or larger than 10 kinds, but the value is {
                    len(n_events)}'
            )

        # --------------------
        # Check the events are not within 1 seconds
        lower_limit = sfreq * 1.0
        sort = np.array(sorted(e[0] for e in events))
        gaps = sort[1:] - sort[:-1]
        min_gap = np.min(gaps)
        if min_gap < lower_limit:
            suspects['min_gap'].append(
                f'The lower limit gap between the events are {
                    lower_limit}({sfreq} Hz), but the value is {min_gap}'
            )

    try:
        _check()
    except Exception:
        suspects['traceback'] = [traceback.format_exc()]

    # --------------------
    # No message is good message
    return suspects, checks


# %% ---- 2024-04-23 ------------------------
# Play ground


# %% ---- 2024-04-23 ------------------------
# Pending


# %% ---- 2024-04-23 ------------------------
# Pending
