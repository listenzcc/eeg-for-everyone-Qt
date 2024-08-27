"""
这几个指标具体指的是。   两个信噪比，反映目标信号强度
SNR_Preprocess，评估预处理前后
SNR_Target，评估目标与非目标

基线漂移度，校正偏移程度
BL_Drift，基线校正数据前数据求标准差

准确率，模型预测准确程度
ACC_ERP，预测标签与真实标签

信息传输率
ITR_ERP，
"""
import numpy as np
import math


def ITR_ERP(N, P, T):
    """
    N: 类别的数量或系统可以区分的目标数（字符集的大小）;
    P: 目标类的概率，即预测的正确概率;
    T: 每次试验的时间间隔（以秒为单位）;
    """
    if P == 0:
        ITR = (1 / T) * (math.log(N, 2) + (1 - P)
                         * math.log((1 - P) / (N - 1), 2))
    elif P == 1:
        ITR = (1 / T) * (math.log(N, 2) + P * math.log(P, 2))
    else:
        ITR = (1 / T) * (math.log(N, 2) + (1 - P) *
                         math.log((1 - P) / (N - 1), 2) + P * math.log(P, 2))
    return 60*ITR


def BL_Drift(correct, channel_names):
    """
    Compute on the data **BEFORE** the preprocessing.

    input
        correct 格式 (stims , channels, length)
        correct为截取时间点前的校正数据
    output
        (channels, 1)
    """

    return dict(zip(channel_names, np.mean(np.std(correct, axis=2), axis=0)))


def SNR_Target(data, label, channel_names):
    """
    calculate snr.
    signal means including target signal(P300 or others)
    noise means  not including target signal
    the variance of the signal and noise is calculated separately
    snr equal signal divide noise.
    input
        signal：(stems, channels, length) 预处理的数据
        label：标签0 or 1 (stems, 1)
        channel_names：各个通道名称

    """
    tar = np.mean(data[np.where(label == 1)[0], :, :], axis=0)
    non_tar = np.mean(data[np.where(label == 0)[0], :, :], axis=0)
    snr_dict = {}
    for montage in range(data.shape[1]):
        single_singal = tar[montage, :]
        single_singal = np.var(single_singal)
        single_noise = non_tar[montage, :]
        single_noise = np.var(single_noise)
        single_snr = single_singal / single_noise

        snr_dict[channel_names[montage]] = single_snr

    return snr_dict


def SNR_Preprocess(noise, signal, label, channel_names):
    """
    input
        noise:(stems, channels, length) 未处理的数据
        signal：(stems, channels, length) 预处理的数据
        label：标签0 or 1 (stems*channels, 1)
        channel_names：各个通道名称

    output
        snr_dict 这里计算的是针对包含P300信号前后数据的信噪比


    """
    signal_tar = np.mean(signal[np.where(label == 1)[0], :, :], axis=0)
    noise_tar = np.mean(noise[np.where(label == 1)[0], :, :], axis=0)

    snr_dict = {}
    for montage in range(noise.shape[1]):
        single_signal = signal_tar[montage, :]
        single_signal = np.var(single_signal)

        single_noise = noise_tar[montage, :]
        single_noise = np.var(single_noise)

        single_snr = single_signal / single_noise

        snr_dict[channel_names[montage]] = single_snr

    return snr_dict


def ACC_ERP(y_true, y_preb):
    """
    input
        y_true (stems, 1)
        y_preb (stems, 1)

    """
    assert y_true.shape[0] == y_preb.shape[0], "真实标签和预测标签的长度不一致"

    y_true = np.array(y_true)
    y_preb = np.array(y_preb)

    correct_predictions = np.sum(y_true == y_preb)

    accuracy = correct_predictions / y_true.shape[0]

    return accuracy
