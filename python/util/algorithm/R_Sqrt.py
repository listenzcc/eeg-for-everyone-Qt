# from Preprocessing.EEGNet_Processing import EEGNet_Processing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import math
import mne


def r2_sqrt(data, events):
    """
    input
        data(stems, channels, length)
        events(stems,1) 0 or 1

    output
        r2(channels, length)
    """
    N1 = len(np.argwhere(events == 0).squeeze())
    N2 = len(np.argwhere(events == 1).squeeze())

    nontar_mean = np.mean(
        data[np.argwhere(events == 0).squeeze(), :, :], axis=0).squeeze()
    tar_mean = np.mean(
        data[np.argwhere(events == 1).squeeze(), :, :], axis=0).squeeze()

    all_std = data.std(axis=0).squeeze()
    r = math.sqrt(N1 * N2) / (N1 + N2) * (tar_mean - nontar_mean) / all_std
    r_sign = np.sign(r)

    r2 = np.square(r) * r_sign

    return r2


def r2_plot(r2, channels_name):
    """
    input
        r2:(channels length) 每个时间点处通道的R平方值
        channels_name 通道名称
    """
    plt.figure(figsize=(10, 6))
    cmap = mcolors.TwoSlopeNorm(
        vmin=r2.min(), vcenter=0, vmax=r2.max())  # 设置颜色映射，使0为白色
    plt.imshow(r2, aspect='auto', cmap='RdBu_r', norm=cmap, origin='lower')
    plt.colorbar(label='R squared value')
    plt.xlabel('Times')
    plt.ylabel('Channels')
    plt.yticks(ticks=np.arange(len(channels_name)), labels=channels_name)
    plt.show()


def r2_topomap_plot(r2, channels_name, time):
    """
    input
        r2:(channels length) 每个时间点处通道的R平方值
        channels_name 通道名称
        time：绘制R平方地形图的时间点 是一个具体数值，请观察峰值时间再填写，例如 95

    """
    # 取出该时间点的R2值
    r2_at_time = r2[:, time-1]

    # 创建info结构，包含通道名称和采样频率（假设为1000Hz）
    info = mne.create_info(channels_name, sfreq=1000, ch_types='eeg')

    # 应用标准蒙太奇
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # 创建mne.EvokedArray对象，用于绘制topomap
    data = np.expand_dims(r2_at_time, axis=1)  # 扩展维度，使其兼容mne的数据格式
    evoked = mne.EvokedArray(data, info)

    # 绘制脑地形图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    mne.viz.plot_topomap(
        evoked.data[:, 0], evoked.info, axes=ax, show=True, cmap='RdBu_r')
