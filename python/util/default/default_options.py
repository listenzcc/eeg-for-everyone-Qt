"""
File: default_options.py
Author: Chuncheng Zhang
Date: 2024-06-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Default options for EEG analysis

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-03 ------------------------
# Requirements and constants


# %% ---- 2024-06-03 ------------------------
# Function and class
class AnyDefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'Any'
    long_name = 'Any protocol'

    # --------------------
    channels = ['C3', 'CZ', 'C4']
    eventIds = ['all']
    epochTimes = dict(tmin=-1.0, tmax=5.0)
    freqBand = dict(freq_l=1.0, freq_h=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why
    epochsKwargs = dict(baseline=(None, 0), decim=10)


class MIDefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'MI'
    long_name = 'Motion Imaging'

    # --------------------
    channels = ['C3', 'CZ', 'C4']
    eventIds = ['240', '241', '242']
    epochTimes = dict(tmin=-1.0, tmax=5.0)
    freqBand = dict(freq_l=1.0, freq_h=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why
    epochsKwargs = dict(baseline=(None, 0), decim=10)


class P300DefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'P300'
    long_name = 'Positive peak on 300ms'

    # --------------------
    channels = [
        'Fz', 'F3', 'F4',
        'Cz', 'C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6',
        'Pz', 'P3', 'P4', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO7', 'PO8',
        'Oz', 'O1', 'O2']
    eventIds = [f'{e}' for e in range(1, 100)]
    epochTimes = dict(tmin=-0.5, tmax=1.0)
    freqBand = dict(freq_l=1.0, freq_h=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why
    epochsKwargs = dict(baseline=(None, 0), decim=10)


# %% ---- 2024-06-03 ------------------------
# Play ground


# %% ---- 2024-06-03 ------------------------
# Pending


# %% ---- 2024-06-03 ------------------------
# Pending
