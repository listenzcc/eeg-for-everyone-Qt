"""
File: analysis.py
Author: Chuncheng Zhang
Date: 2025-05-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Convert epo.fif into bdf file.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-13 ------------------------
# Requirements and constants
from datetime import datetime
from mne.time_frequency import psd_array_welch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import mne
from pathlib import Path
from matplotlib.font_manager import FontProperties

# 指定字体文件路径
font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径
myfont = FontProperties(fname=font_path, size=12)
plt.rcParams['font.family'] = 'SIMHEI'  # 'Arial'  # 使用无衬线字体更专业
plt.rcParams['text.color'] = '#333333'  # 深灰色文字更易读
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')
# %% ---- 2025-05-13 ------------------------
# Function and class
# 从epochs对象提取信息


def get_recording_info(epochs):
    """从epochs对象提取记录信息"""
    info = {
        'model': epochs.info.get('device_info', {}),
        'sfreq': f"{epochs.info['sfreq']} Hz",
        'n_channels': len(epochs.ch_names),
        'montage': epochs.info.get('dig', '未知布局'),
        'highpass': epochs.info.get('highpass', '未知'),
        'lowpass': epochs.info.get('lowpass', '未知')
    }
    return info


def get_analysis_info(epochs):
    """从epochs对象提取分析参数"""
    info = {
        'epoch_length': f"{epochs.tmax - epochs.tmin:.2f} 秒",
        'baseline': str(epochs.baseline),
        'reject': '已应用' if hasattr(epochs, 'reject') else '未应用',
        'n_epochs': len(epochs),
        'events': ', '.join([f"{k}({v})" for k, v in epochs.event_id.items()])
    }
    return info


# %% ---- 2025-05-13 ------------------------
# Play ground
# 1. 加载数据
epochs = mne.read_epochs('S001_epo.fif')
print(epochs)
print(epochs.events)
print(epochs.event_id)
print(epochs.ch_names)

# %% ---- 2025-05-13 ------------------------
# Pending

# %% ---- 2025-05-13 ------------------------
# Pending


def first_page(pdf):
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')

    # 生成报告文本
    recording_info = get_recording_info(epochs)
    analysis_info = get_analysis_info(epochs)

    # 分两列排版
    left_text = f"""
    数据采集信息
    - 设备型号: {recording_info['model']}
    - 采样率: {recording_info['sfreq']}
    - 电极数量: {recording_info['n_channels']} 通道
    - 电极布局: {recording_info['montage']}
    - 滤波设置: {recording_info['highpass']}-{recording_info['lowpass']} Hz带通
    """

    right_text = f"""
    分析参数说明
    - 分段长度: {analysis_info['epoch_length']}
    - 基线校正: {analysis_info['baseline']}
    - 伪迹剔除: {analysis_info['reject']}
    - 有效Epoch数量: {analysis_info['n_epochs']}
    - 事件类型: {analysis_info['events']}
    - 频段功率: 对数转换
    """

    plt.text(0.05, 0.9, "EEG数据分析报告说明", fontsize=16, fontweight='bold')
    plt.text(0.05, 0.5, left_text, fontsize=12, linespacing=1.8)
    plt.text(0.55, 0.5, right_text, fontsize=12, linespacing=1.8)

    # 添加分割线
    plt.axvline(x=0.5, ymin=0.1, ymax=0.9, color='lightgray', linestyle='--')

    pdf.savefig()
    plt.close()


# 2. 创建PDF报告
with PdfPages('EEG_Analysis_Report.pdf') as pdf:
    first_page(pdf)

    plt.figure(figsize=(11, 8))

    # 3. 绘制所有epoch的波形图
    plt.suptitle('EEG Waveforms - All Epochs', fontsize=16)
    for i, epoch in enumerate(epochs.get_data()):
        plt.subplot(len(epochs), 1, i+1)
        plt.plot(epochs.times, epoch.T,
                 label=f'Epoch {i+1} (Event ID: {epochs.events[i, 2]})')
        plt.ylabel('Amplitude (μV)')
        plt.legend(loc='lower right')
        if i == len(epochs)-1:
            plt.xlabel('Time (s)')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 4. 绘制各epoch叠加波形图
    plt.figure(figsize=(11, 6))
    for i, epoch in enumerate(epochs.get_data()):
        plt.plot(epochs.times, epoch.T, alpha=0.7,
                 label=f'Epoch {i+1} (Event ID: {epochs.events[i, 2]})')
    plt.title('Superimposed EEG Waveforms')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # 5. 频谱分析 - 各经典频段
    freq_bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-45 Hz)': (30, 45)
    }

    # 计算功率谱密度 - 使用新的API
    sfreq = epochs.info['sfreq']
    psds, freqs = psd_array_welch(
        epochs.get_data(),  # 输入数据 (n_epochs, n_channels, n_times)
        sfreq=sfreq,
        fmin=0.5,
        fmax=45,
        n_fft=1024,  # 增加FFT点数，提高频率分辨率
        n_overlap=512,  # 相应增加重叠点数
        verbose=False
    )
    print(freqs)

    # 绘制总频谱图
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, 10 * np.log10(psds.mean(axis=(0, 1))), color='black')
    plt.title('Power Spectral Density (All Channels)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')

    # 标记各频段
    # colors = ['blue', 'green', 'red', 'purple', 'orange']
    colors = [
        '#66C2A5',  # 蓝绿
        '#FC8D62',  # 橙红
        '#8DA0CB',  # 紫蓝
        '#E78AC3',  # 粉红
        '#A6D854'   # 黄绿
    ]
    for (band, (fmin, fmax)), color in zip(freq_bands.items(), colors):
        mask = (freqs >= fmin) & (freqs <= fmax)
        plt.fill_between(freqs[mask], 10 * np.log10(psds.mean(axis=(0, 1)))[mask],
                         color=color, alpha=0.3, label=band)
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # 6. 各频段功率分布
    plt.figure(figsize=(11, 6))
    band_powers = []
    band_names = []

    for band, (fmin, fmax) in freq_bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        power = psds[:, :, mask].mean(axis=(0, 2))
        band_powers.append(power)
        band_names.append(band)
        print(band, power)

    plt.yscale('log')
    plt.bar(band_names, [bp[0] for bp in band_powers], color=colors)
    plt.title('Relative Power in Different Frequency Bands')
    plt.ylabel('Power (uV2/Hz)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    pdf.savefig()
    plt.close()

    # 7. 各epoch的频段功率比较
    plt.figure(figsize=(11, 8))
    for i, epoch_num in enumerate(range(len(epochs))):
        plt.subplot(len(epochs), 1, i+1)
        epoch_powers = []
        for band, (fmin, fmax) in freq_bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = psds[epoch_num, :, mask].mean(axis=(0, 1))
            epoch_powers.append(power)

        plt.yscale('log')
        plt.bar(band_names, epoch_powers, color=colors)
        plt.title(
            f'Epoch {epoch_num+1} (Event ID: {epochs.events[epoch_num, 2]}) - Band Power')
        plt.ylabel('Power (uV2/Hz)')
        # plt.xticks(rotation=45)
        if i == len(epochs)-1:
            plt.xlabel('Frequency Band')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("分析报告已保存为 EEG_Analysis_Report.pdf")
