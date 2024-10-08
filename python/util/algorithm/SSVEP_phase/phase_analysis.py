import numpy as np


def FFT_Analysis_N(data, fs, N):
    '''
    fre, y, p = FFT_Analysis_N(data, fs, N)
    该函数利用快速傅里叶变换（FFT）对输入信号进行频域分析，计算其幅度谱和相位谱

    输入参数：
    - data (np.ndarray):  输入信号的时间序列，维度为(timepoint,)
    - fs (int): 采样频率
    - N (int): 输入信号的长度

    输出：
    - fre (np.ndarray): 频率数组，表示对应FFT结果的频率（单位为赫兹），包含从0到Nyquist频率（采样频率的一半）的频率值，维度为(timepoint/2,)
    - y (np.ndarray): 幅度谱，表示每个频率成分的幅度，维度为(timepoint/2,)
    - p (np.ndarray): 相位谱，表示每个频率成分的相位，维度为(timepoint/2,)
    '''
    N = len(data)
    n = np.arange(N)
    t = n / fs
    x = np.fft.fft(data, N)
    mag = np.abs(x)
    tol = 1e-9
    x[np.abs(x) < tol] = 0
    p = np.angle(x[:N // 2]) / np.pi
    f = n * fs / N
    fre = f[:N // 2]
    y = mag[:N // 2] * 2 / N
    return fre, y, p


def compute_phase_diff(data, sim_freq, sampling_rate):
    '''
    绘制相位图和相位差图
    p_1, dp = phase_plot(data, sim_freq, sampling_rate)
    该函数遍历频率，提取匹配数据并计算其均值执行FFT分析，提取相位信息并计算相邻频率之间的相位差

    输入参数：
    - data (np.ndarray): 输入数据，维度为(trials,channels,time point)
    - sim_freq (np.ndarray): 输入数据的频率，维度为(trials,)，trials与data维度一致
    - sampling rate: (int): 采样频率

    输出：
    - p_1 (np.ndarray): 对应于每个频率的相位信息，维度为(freq_num,)
    - dp (np.ndarray): 相位差，表示相邻频率之间的相位变化，维度为(freq_num-1,)
    '''
    # Compute freqs
    # - freqs (list): 待分析的频率列表
    sim_freq = np.array(sim_freq).flatten()
    freqs = sorted(np.unique(sim_freq))
    print('freqs', freqs)

    freq_sort = np.sort(freqs)
    freq_index = np.argsort(freqs)
    Npoints = round(sampling_rate / np.min(np.diff(freq_sort)))

    data_all = np.zeros((data.shape[2], len(freqs)))
    p_1 = np.zeros(len(freqs))
    for ii in range(len(freqs)):
        current_freq_index = freq_index[ii]

        matching_types = np.where(sim_freq == freqs[current_freq_index])[0]
        print(ii, matching_types)

        if matching_types.size > 0:
            selected_data = data[matching_types, :, :]
            data_mean = np.mean(selected_data, axis=0)
            data_mean2 = np.mean(data_mean, axis=0)
            data_all[:, ii] = data_mean2.flatten()
            fre, y, p = FFT_Analysis_N(data_mean2, sampling_rate, Npoints)
            b = np.where(np.abs(fre - freq_sort[ii]) < 0.01)[0]
            if b.size > 0:
                p_1[ii] = p[b[0]]

    dp = np.diff(p_1)
    dp[dp < -1] += 2

    return p_1, dp
