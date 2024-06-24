# %%
from scipy.io import loadmat
from scipy import signal
import numpy as np
from numpy import *
from scipy.stats import pearsonr
from CCA import CCA
from FBCCA import FBCCA

# %%
# 陷波


def get_pre_filter(samp_rate):
    fs = samp_rate
    f0 = 50
    q = 20  # 凝胶电极需要稍微降低品质因数
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    return b, a

# 滤波器组


def get_filterbank(samp_rate, lowpassband, highpassband):
    # SSVEP
    fs = samp_rate / 2
    A = []
    B = []
    # lowpassband = np.linspace(6, 54, 7)
    for i in range(0, len(lowpassband)):
        fl = lowpassband[i]
        fh = highpassband[i]
        N, Wn = signal.ellipord(
            [fl / fs, fh / fs], [(fl - 4) / fs, (fh + 10) / fs], 3, 40)
        b, a = signal.ellip(N, 1, 40, Wn, 'bandpass')
        A.append(a)
        B.append(b)
    return B, A

# 构建正余弦模板，输入采样率（Hz），刺激频率（Hz），倍频数量，刺激时长（秒）


def get_template(samp_rate, stim_event_freq, multiple_freq, stim_time):
    target_template_set = []
    cal_len = stim_time * samp_rate
    samp_point = np.linspace(
        0, (cal_len - 1) / samp_rate, int(cal_len), endpoint=True)
    # (1 * 计算长度)的二维矩阵
    samp_point = samp_point.reshape(1, len(samp_point))
    for freq in stim_event_freq:
        # 基频 + 倍频
        test_freq = np.linspace(
            freq, freq * multiple_freq, int(multiple_freq), endpoint=True)
        # (1 * 倍频数量)的二维矩阵
        test_freq = test_freq.reshape(1, len(test_freq))
        # (倍频数量 * 计算长度)的二维矩阵
        num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
        cos_set = np.cos(num_matrix)
        sin_set = np.sin(num_matrix)
        cs_set = np.append(cos_set, sin_set, axis=0)
        target_template_set.append(cs_set)
    return target_template_set


# %%
samp_rate = 250

lowpassband = [6, 14, 22, 30, 38, 46, 54]
highpassband = [90, 90, 90, 90, 90, 90, 90]
B, A = get_filterbank(samp_rate, lowpassband, highpassband)
nFilterBank = len(lowpassband)

# 滤波器组加权参数
a = -1.25
b = 0.25
n = arange(nFilterBank)+1
W = np.power(n, (-a))+b
# 使用通道
usechan = [47, 53, 54, 55, 56, 57, 60, 61, 62]
# 刺激频率
# stim_event_freq = [round(8.0 + i * 0.2, 1) for i in range(40)]
stim_event_freq = [round(j+8 + i * 0.2, 1) for i in range(5) for j in range(8)]
# 刺激时间（秒）
stim_time = 4
cal_len = stim_time * samp_rate
# 倍频数
multiple_freq = 5
# 构建正余弦模板

target_template_set = get_template(
    samp_rate, stim_event_freq, multiple_freq, stim_time)

FBCCA_method = FBCCA(target_template_set, a, b, nFilterBank)
CCA_method = CCA(target_template_set)

# 读数据，示例数据每个试次时长6秒，其中0.5秒到5.5秒为视觉刺激
data = loadmat("./test_data.mat")
Data = data["data"]
Data_raw = Data[usechan, :, :]  # 通道数 * 采样点 * 试次

Type = np.zeros([1, 40])

# 直接用FBCCA或CCA分类
for trial in range(np.size(Data_raw, 2)):

    Data_1 = Data_raw[:, :, trial]
    Data_epoch = []
    p = np.zeros([len(target_template_set), nFilterBank])
    for i in range(nFilterBank):
        Data_filtered = (signal.filtfilt(B[i], A[i], Data_1, axis=1))
        Data_epoch.append(Data_filtered[:, 160:160+cal_len])

    Type[0, trial] = FBCCA_method.predict(Data_epoch)
    # Type[0, trial] = CCA_method.predict(Data_epoch[0])
    print("label: %d result is %d" % (trial, Type[0, trial]))

acc = np.sum(Type == arange(40))/40

mean_acc = np.mean(acc)
print("Accuracy is %f" % (mean_acc))

# %%
print(np.array(Data_epoch).shape)

# %%
# %%

# 按步骤计算FBCCA，用CCA.fit()
# for trial in range(np.size(Data_raw,2)):
#
#     Data_1 = Data_raw[:, :, trial]
#     p = np.zeros([len(target_template_set), nFilterBank])
#     for i in range(nFilterBank):
#         Data_filtered = (signal.filtfilt(B[i], A[i], Data_1, axis=1))
#         Data_epoch = Data_filtered[:, 160:160+cal_len]
#         for j in range(0, len(target_template_set)):
#             template = target_template_set[j]
#             CCA_A, CCA_B, rho = CCA_method.fit(Data_epoch, template)
#             c = pearsonr(np.dot(Data_epoch.T, CCA_A[:, 0]), np.dot(template.T, CCA_B[:, 0]))
#             # c[0]与rho[0]相等
#             p[j, i] = pow(c[0], 2) * W[i]
#             # p[j, i] = pow(rho[0], 2) * W[i]
#     pFB = np.sum(p, axis=1)
#     # Type[0,trial] = FBCCA_method.recognize(Data_epoch)
#     Type[0, trial] = pFB.argmax()
#     print("label: %d result is %d" %(trial,Type[0,trial]))
#
# acc=np.sum(Type==arange(40))/40
#
# mean_acc=np.mean(acc)
# print("Accuracy is %f" %(mean_acc))


# 按步骤计算FBCCA，用FBCCA.fit
# for trial in range(np.size(Data_raw,2)):
#
#     Data_1 = Data_raw[:, :, trial]
#     p = np.zeros([len(target_template_set), nFilterBank])
#     Data_epoch = []
#     for i in range(nFilterBank):
#         Data_filtered = (signal.filtfilt(B[i], A[i], Data_1, axis=1))
#         Data_epoch.append(Data_filtered[:, 160:160+cal_len])
#
#     for j in range(0, len(target_template_set)):
#         template = target_template_set[j]
#         FBCCA_A, FBCCA_B, rho = FBCCA_method.fit(Data_epoch, template)
#
#         for i in range(nFilterBank):
#             c = pearsonr(np.dot(Data_epoch[i].T, FBCCA_A[i][:, 0]), np.dot(template.T, FBCCA_B[i][:, 0]))
#             # c[0]与rho[i][0]相等
#             p[j, i] = pow(c[0], 2) * W[i]
#             # p[j, i] = pow(rho[i][0], 2) * W[i]
#     pFB = np.sum(p, axis=1)
#     # Type[0,trial] = FBCCA_method.recognize(Data_epoch)
#     Type[0, trial] = pFB.argmax()
#     print("label: %d result is %d" %(trial,Type[0,trial]))
#
# acc=np.sum(Type==arange(40))/40
#
# mean_acc=np.mean(acc)
# print("Accuracy is %f" %(mean_acc))
