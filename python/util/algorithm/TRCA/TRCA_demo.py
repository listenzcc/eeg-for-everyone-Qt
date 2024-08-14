from scipy.io import loadmat
from scipy import signal
import numpy as np
from numpy import *
from TRCA import TRCA
from cal_itr import cal_itr
# 陷波
def get_pre_filter(samp_rate):
    fs = samp_rate
    f0 = 50
    q = 20 # 凝胶电极需要稍微降低品质因数
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    return b, a

# 滤波器组
def get_filter(samp_rate, lowpassband, highpassband):

    fs = samp_rate / 2
    fl = lowpassband
    fh = highpassband
    N, Wn = signal.ellipord([fl / fs, fh / fs], [(fl - 4) / fs, (fh + 10) / fs], 3, 40)
    B, A = signal.ellip(N, 1, 40, Wn, 'bandpass')

    return B, A

samp_rate=250
lowpassband = 6
highpassband = 90
B, A = get_filter(samp_rate, lowpassband, highpassband)

# 刺激频率
# stim_event_freq = [round(8.0 + i * 0.2, 1) for i in range(40)]
stim_event_freq = [round(j+8 + i * 0.2, 1) for i in range(5) for j in range(8)]
# 刺激时间（秒）
stim_time = 1
cal_len = round(stim_time * samp_rate)

# 读数据，示例数据每个试次时长6秒，其中0.5秒到5.5秒为视觉刺激
data = loadmat("./test_data.mat")
Data_raw = data["data"]
type = data["type"]

Data_epoch = np.zeros([240, 9, cal_len]) # trials*channels*timepoint
# 滤波
for trial in range(np.size(Data_raw,2)):
    Data_1 = Data_raw[:, :, trial]
    Data_filtered = (signal.filtfilt(B, A, Data_1, axis=1))
    Data_epoch[trial,:,:] = Data_filtered[:, 160:160+cal_len]
# 分训练测试数据
Data_train = Data_epoch[0:200,:,:]
Type_train = type[0][0:200]
Data_test = Data_epoch[200:240,:,:]
Type_test = type[0][200:240]
Type_num = 40
# 初始化
trca_method = TRCA(Type_num)
# fit
model = trca_method.fit(Data_train, Type_train)
# predict
result = trca_method.predict(model, Data_test)
score = trca_method.predict_proba(model, Data_test)
# 准确率
acc = np.sum(Type_test==result)/len(Type_test)
# ITR 假设视线转移时间为0.5秒
itr = cal_itr(stim_time + 0.5, Type_num, acc)
print("Accuracy is %f" %(acc))
print("ITR is %f bits/min" %(itr))
