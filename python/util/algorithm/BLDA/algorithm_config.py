class AlgorithmConfig:
    """
    文件包含可修改的参数配置
    """
    # 通道名称，如要调整通道，请保证命名与该矩阵一致，避免报错。
    ALL_CHANNEL = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2',
                   'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3',
                   'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4',
                   'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                   'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
                   'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG',
                   'HEOR', 'HEOL', 'VEOU', 'VEOL']

    # 定义采样率，此矩阵勿动。
    SAMPLE_RATE = [250, 250]

    # 选择导联名称,注意导联名称得在上面的那个全集里面,这里用了21导
    SELECT_CHANNEL_NAME = ['Fz', 'Cz', 'C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6',
                           'Pz', 'P3', 'P4', 'P7', 'P8', 'POz', 'PO3', 'PO4',
                           'PO7', 'PO8', 'Oz', 'O1', 'O2']


    # 计算时间,指取多长的时间长度
    cal_time = 0.6

    # 计算长度
    CAL_LEN: int = round(cal_time * SAMPLE_RATE[0])

    # 下采样程度，此处勿动
    DOWN_SAMPLE_RATE = SAMPLE_RATE[0] // SAMPLE_RATE[1]

    # 滤波器参数
    FILTER_PARAM = dict(sample_frequency=SAMPLE_RATE[0], notch_f=50, band_f_up=0.5, band_f_down=20)

    # 交叉验证次数
    CV_NUM = 5

    # 保存的模型的名称
    METHOD_NAME = 'trained_model'

    # 是否需要做重参考
    RE_SAMP = ['T7', 'T8']
    # self.re_samp = []


if __name__ == '__main__':
    print("此处是参数设置")
