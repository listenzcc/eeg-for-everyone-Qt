from AlgorithmImplement.Interface.AlgorithmInterface import AlgorithmInterface
# from Algorithm.Interface.Model.ReportModel import ReportModel
# from AlgorithmSystem.AlgorithmImplement.SSVEP.CCA import CCA
from AlgorithmImplement.SSVEP.FBCCA import FBCCA
from scipy import signal
import numpy as np
import math



class AlgorithmImplementSSVEP(AlgorithmInterface):
    # 类属性：范式名称
    PARADIGMNAME = 'SSVEP'
    # PARADIGMNAME = 'cVEP'

    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        samp_rate = 250
        # 选择导联编号
        # self.select_channel = [50,51,52,53,54,57,58,59]
        self.select_channel = [1, 2, 3, 4, 5, 6, 7, 8]	
        self.select_channel = [i - 1 for i in self.select_channel]
        # trial开始trigger，题目说明中给出
        self.trial_start_trig = 240
        # 倍频数
        multiple_freq = 5
        # 计算时间
        cal_time = 4
        # 计算偏移时间（s）
        offset_time = 0.14
        # 偏移长度
        self.offset_len = math.floor(offset_time * samp_rate)
        # 计算长度
        self.cal_len = cal_time * samp_rate
        # 频率集合
        stim_event_freq = [round(8.0 + i * 0.2, 1) for i in range(40)]
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        #
        self.filterBankB, self.filterBankA = self.__get_filterbank(samp_rate)
        # 正余弦参考信号
        target_template_set = []

        if self.PARADIGMNAME == 'SSVEP':
            # 采样点
            samp_point = np.linspace(0, (self.cal_len - 1) / samp_rate, int(self.cal_len), endpoint=True)
            # (1 * 计算长度)的二维矩阵
            samp_point = samp_point.reshape(1, len(samp_point))
            # 对于每个频率
            for freq in stim_event_freq:
                # 基频 + 倍频
                test_freq = np.linspace(freq, freq * multiple_freq, int(multiple_freq), endpoint=True)
                # (1 * 倍频数量)的二维矩阵
                test_freq = test_freq.reshape(1, len(test_freq))
                # (倍频数量 * 计算长度)的二维矩阵
                num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
                cos_set = np.cos(num_matrix)
                sin_set = np.sin(num_matrix)
                cs_set = np.append(cos_set, sin_set, axis=0)
                target_template_set.append(cs_set)
        else:
            # cVEP stimulation sequence
            seq = np.load('./sequenceCVEP.npy')
            print(seq.shape[0])
            for i in range(seq.shape[0]):
                target_template_set.append(seq[i])
                
        # 初始化算法
        # self.method = CCA(target_template_set)
        self.method = FBCCA(target_template_set)

    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        while not end_flag:
            data_model = self.algo_sys_mng.get_data()
            if data_model is None:
                continue
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)
            else:
                # 计算模式，则进行处理
                cal_flag, result = self.__cal_proc(data_model)
                # 如果有结果，则进行报告
                if result is not None:
                    self.algo_sys_mng.report(result)
                    # 清空缓存
                    self.__clear_cache()
            end_flag = data_model.finish_flag


    def __idle_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_start_trig)[0]
        # 脑电数据
        eeg_data = data[0: -1, :]
        if len(trigger_idx) > 0:
            # 有trial开始trigger则进行计算
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos: eeg_data.shape[1]]
        else:
            # 没有trial开始trigger则
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_start_trig)[0]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 如果trigger为空，表示依然在当前试次中，根据数据长度判断是否计算
        if len(trigger_idx) == 0:
            # 当已缓存的数据大于等于所需要使用的计算数据时，进行计算
            if self.cache_data.shape[1] >= self.cal_len:
                # 获取所需计算长度的数据
                self.cache_data = self.cache_data[:, 0: int(self.cal_len)]
                # 考虑偏移量
                use_data = self.cache_data[:, self.offset_len: self.cache_data.shape[1]]
                # 滤波处理
                use_data = self.__preprocess(use_data)
                use_data_FB = self.__preprocessFB(use_data)
                # 开始计算，返回计算结果
                # result = self.method.recognize(use_data)
                result = self.method.recognize(use_data_FB)
                # 停止计算模式
                cal_flag = False
            else:
                # 反之继续采集数据
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                result = None
                cal_flag = True
        # 下一试次已经开始,需要强制结束计算
        else:
            # 下一个trial开始trigger的位置
            next_trial_start_trig_pos = trigger_idx[0]
            # 如果拼接该数据包中部分的数据后，可以满足所需要的计算长度，则拼接数据达到所需要的计算长度
            # 如果拼接完该trial的所有数据后仍无法满足所需要的数据长度，则只能使用该trial的全部数据进行计算
            use_len = min(next_trial_start_trig_pos, self.cal_len - self.cache_data.shape[1])
            self.cache_data = np.append(self.cache_data, eeg_data[:, 0: use_len], axis=1)
            # 考虑偏移量
            use_data = self.cache_data[:, self.offset_len: self.cache_data.shape[1]]
            # 滤波处理
            use_data = self.__preprocess(use_data)
            use_data_FB = self.__preprocessFB(use_data)
            # 开始计算
            # result = self.method.recognize(use_data)
            result = self.method.recognize(use_data_FB)
            # 开始新试次的计算模式
            cal_flag = True
            # 清除缓存的数据
            self.__clear_cache()
            # 添加新试次数据
            self.cache_data = eeg_data[:, next_trial_start_trig_pos: eeg_data.shape[1]]
        return cal_flag, result

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 20 # 凝胶电极需要稍微降低品质因数
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __get_filterbank(self, samp_rate):
        # SSVEP
        fs = samp_rate / 2
        A = []
        B = []
        lowpassband = np.linspace(6, 54, 7)
        for i in range(0, len(lowpassband)):
            fl = lowpassband[i]
            N, Wn = signal.ellipord([fl / fs, 90 / fs], [(fl - 4) / fs, 100 / fs], 3, 40)
            b, a = signal.ellip(N, 1, 40, Wn, 'bandpass')
            A.append(a)
            B.append(b)
        return B, A

    # def __get_filterbank(self, samp_rate):
    #     # cVEP
    #     fs = samp_rate / 2
    #     A = []
    #     B = []
    #     lowpassband = np.linspace(15, 30, 2)
    #     # print(lowpassband)
    #     for i in range(0, len(lowpassband)):
    #         fl = lowpassband[i]
    #         N, Wn = signal.ellipord([fl / fs, 80 / fs], [(fl - 2) / fs, 90 / fs], 3, 40)
    #         b, a = signal.ellip(N, 1, 40, Wn, 'bandpass')
    #         A.append(a)
    #         B.append(b)
    #     return B, A

    def __clear_cache(self):
        self.cache_data = None

    def __preprocess(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        filter_data = signal.filtfilt(self.filterB, self.filterA, data)
        return filter_data

    def __preprocessFB(self, data):
        filter_data=[]
        # 选择使用的导联
        # data = data[self.select_channel, :]
        for i in range(0,len(self.filterBankB)):
            filter_data_temp = signal.filtfilt(self.filterBankB[i], self.filterBankA[i], data)
            filter_data.append(filter_data_temp)
        return filter_data