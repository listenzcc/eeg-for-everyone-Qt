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
    freqBand = dict(l_freq=1.0, h_freq=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why

    # --------------------
    epochsKwargs = dict(
        baseline=(None, 0),
        detrend=1,
        decim=10,
        event_repeated='drop',
    )

    # --------------------
    otherOptions = dict(
        ref_channels=[],
    )


class MIDefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'MI'
    long_name = 'MI experiment from Bei Wang'

    # --------------------
    channels = ['C3', 'CZ', 'C4']
    eventIds = ['240', '241', '242']
    epochTimes = dict(tmin=-1.0, tmax=5.0)
    freqBand = dict(l_freq=1.0, h_freq=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why

    # --------------------
    epochsKwargs = dict(
        baseline=(None, 0),
        detrend=1,
        decim=10,
        event_repeated='drop'
    )

    # --------------------
    otherOptions = dict(
        ref_channels=[],
    )


class P300DefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'P300'
    long_name = 'P300 experiment from Bei Wang'

    # --------------------
    channels = ['Fpz', 'Fp1', 'Fp2',
                'AF3', 'AF4', 'AF7', 'AF8',
                'Fz', 'F3', 'F4', 'F7', 'F8',
                'FCz', 'FC3', 'FC4', 'FT7', 'FT8',
                'Cz', 'C3', 'C4', 'T7', 'T8',
                'CP3', 'CP4', 'TP7', 'TP8',
                'Pz', 'P3', 'P4', 'P7', 'P8',
                'POz', 'PO3', 'PO4', 'PO7', 'PO8',
                'Oz', 'O1', 'O2']
    eventIds = [f'{e}' for e in range(1, 100)]
    epochTimes = dict(tmin=-0.5, tmax=1.5)
    freqBand = dict(l_freq=1.0, h_freq=25.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why

    # --------------------
    epochsKwargs = dict(
        baseline=(None, 0),
        detrend=1,
        decim=10,
        event_repeated='drop',
    )

    # --------------------
    otherOptions = dict(
        ref_channels=[],
    )


class SSVEPDefaultOptions:
    """
    The default options for MI experiment protocol.
    ! The attributes are type specific.
    """

    # --------------------
    short_name = 'SSVEP'
    long_name = 'SSVEP experiment from Li Zheng'

    # --------------------
    channels = ['PO3', 'PO5', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']
    eventIds = [f'{e}' for e in range(1, 241)]
    epochTimes = dict(tmin=-0.2, tmax=1.0)
    freqBand = dict(l_freq=3.0, h_freq=90.0)
    reject = dict(eeg=0.4)  # It is very large, and I don't know why

    # --------------------
    epochsKwargs = dict(
        baseline=(None, 0),
        detrend=1,
        decim=4,
        event_repeated='drop',
    )

    # --------------------
    otherOptions = dict(
        ref_channels=[],
    )

# %% ---- 2024-06-03 ------------------------
# Play ground


# %% ---- 2024-06-03 ------------------------
# Pending


# %% ---- 2024-06-03 ------------------------
# Pending
