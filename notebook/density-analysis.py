"""
File: density-analysis.py
Author: Chuncheng Zhang
Date: 2024-07-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute large data density analysis for P300 dataset.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-07-03 ------------------------
# Requirements and constants
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from rich import print
from IPython.display import display

# Use software's reading method
p = Path(__file__).parent.parent  # noqa
sys.path.append(p.as_posix())  # noqa

from python.util.load_data.load_epochs import EpochsObject
from python.util.default.default_options import P300DefaultOptions


# %% ---- 2024-07-03 ------------------------
# Function and class


def find_files():
    cache_folder = Path(__file__).parent.parent.joinpath("cache")

    found_files = pd.read_pickle(cache_folder.joinpath("found_files"))
    check_results = pd.read_pickle(cache_folder.joinpath("check_results"))

    found = pd.merge(check_results, found_files, on="path")
    found = found[found["status"] != "failed"]
    return found[found["protocol"] == "P300(3X3)"]


def read_file(se: pd.Series):
    epochs_obj = EpochsObject(se)

    kwargs = {}
    kwargs |= dict(epochsKwargs=P300DefaultOptions.epochsKwargs)
    kwargs |= dict(eventIds=P300DefaultOptions.eventIds)
    kwargs |= dict(epochTimes=dict(tmin=-0.5, tmax=1.0))

    epochs_obj.get_epochs(kwargs)
    epochs = epochs_obj.epochs

    # nTrials x nChannels x nTimes
    data = epochs.get_data()
    times = epochs.times
    events = epochs.events
    ch_names = epochs.ch_names

    # Select channel[ch] and Squeeze to nTrials x nTimes
    ch = "Oz"
    data = data[:, [e == ch for e in ch_names]].squeeze()

    # Only use the last column of the events
    events = events[:, -1]

    # Change unit to uv(1e-6)
    data *= 1e6

    return dict(
        epochs_obj=epochs_obj,
        epochs=epochs,
        data=data,
        times=times,
        events=events,
        ch_names=ch_names,
    )


class DensityWorkload:
    vmin = -10
    vmax = 15
    nbins = 100
    bins_center = np.linspace(vmin, vmax, nbins)
    sigma = bins_center[1] - bins_center[0]
    ntimes = 151  # it equals to ntimes of data

    def __init__(self, ntimes: int = None):
        if ntimes:
            self.ntimes = ntimes

        self.score_buffer = np.zeros((self.ntimes, self.nbins))
        print("score_buffer", self.score_buffer.shape)

    def compute(self, ts: np.ndarray, update: bool = True):
        data = np.vstack([ts] * self.nbins).T
        diff = data - self.bins_center
        num = -np.power(diff, 2)
        den = 2 * np.power(self.sigma, 2)
        score = np.exp(num / den)
        if update:
            self.score_buffer += score
        return score


found_files = find_files()
display(found_files)

# %% ---- 2024-07-03 ------------------------
# Play ground
obj = read_file(found_files.iloc[0])
data = obj["data"]
times = obj["times"]
epochs = obj["epochs"]
events = obj["events"]
data2 = data[events == 2]
print("data", data.shape)
print("data2", data2.shape)
print("epochs", epochs)
print("events", events.shape)

dw = DensityWorkload(ntimes=len(times))

# ! Start loop with n times
n = 100

for i in tqdm(range(n), "Computing"):
    obj = read_file(found_files.iloc[i * 2])
    data = obj["data"]
    events = obj["events"]
    data2 = data[events == 2]

    for ts in data2:
        score = dw.compute(ts)

    print("score", score.shape)


# %% ---- 2024-07-03 ------------------------
# Pending
cmap = sns.cubehelix_palette(start=2.9, light=0.9, as_cmap=True, reverse=True)
cmap = cmap.with_extremes(bad=cmap(0))
print(cmap)

# ntimes x nbins
score_map = dw.score_buffer.copy()
for e in score_map:
    e /= np.sum(e)
print(score_map.shape)

df = pd.DataFrame(score_map.T, columns=times)
df.index = [f"{e:0.2f}" for e in dw.bins_center]
fig, ax = plt.subplots(1, 1)
sns.heatmap(df, ax=ax, cmap=cmap)
ax.invert_yaxis()
plt.show()

# %%
# %%
time_map = [e > 0.1 and e < 0.5 for e in times]

data1 = data[events == 1]
print(data1.shape)

# %%


def valuate(sd, class_name):
    scores = []
    for ts in tqdm(sd, f"Computing prob ({class_name})"):
        d = np.array([
            dw.compute(np.array([e]), update=False)
            for e in ts[time_map]])
        scores.append(np.sum(d.squeeze() * score_map[time_map]))

    score_array = np.array(scores)
    print(score_array.shape)

    df = pd.DataFrame(score_array, columns=["score"])
    df["class"] = class_name
    return df


df1 = valuate(data1, "d1")
df2 = valuate(data2, "d2")

df = pd.concat([df1, df2])
display(df)

fig, ax = plt.subplots(1, 1)
sns.violinplot(df, x="class", y="score", hue="class", ax=ax)
plt.show()

# %% ---- 2024-07-03 ------------------------
# Pending
