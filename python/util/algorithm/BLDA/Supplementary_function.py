# import os
# import pickle
# import random
import numpy as np
# from mne import io
from .readbdfdata import readbdfdata


# import math


def load_data(folder_path):
    # 这里存在的问题就是在线部分为了减轻数据包的体量，刚传进来就下采样了。
    # 在这里就先放成250Hz的固定值
    down_sample_rate = 1000 // 250
    filename = ['data.bdf', 'evt.bdf']
    down_sample_data = dict(data=[], evt=[], ch_name=[])

    eeg = readbdfdata(filename, folder_path)

    # 下面的六行是加入了下采样
    data = eeg['data']
    evt = eeg['events'][:, [0, 2]]
    # 单点采样法
    down_sample_data['data'] = data[:, ::down_sample_rate]
    evt_idx = evt[:, 0] // down_sample_rate
    evt[:, 0] = evt_idx
    down_sample_data['evt'] = evt

    # down_sample_data['data'] = eeg['data'][:59, :]
    # down_sample_data['evt'] = eeg['events'][:, [0, 2]]

    down_sample_data['ch_name'] = eeg['ch_names']

    return down_sample_data


def channel_name_to_index(chn_list, selected_list):
    """
    根据提供的通道名称（如 [T7, T8]）, 给出通道的布尔索引

    inputs
    chn_list: list of channel names
    selected_list: list of selected channel names

    outputs
    chn_idx: numpy bool array of selected channels
    non_chn_idx: numpy bool array of non_selected channels
    """
    ch_idx = np.zeros(len(chn_list))
    for i, ch_name in enumerate(chn_list):
        if ch_name in selected_list:
            ch_idx[i] = 1
    assert np.sum(ch_idx) == len(selected_list), "提供的列表中的部分通道无法找到"
    return ch_idx == 1, ch_idx == 0


def cal_acc(pred, target):
    assert len(pred) == len(target)
    acc = np.sum(pred == target) / len(pred)
    return acc


def cal_f(pred, target):
    assert len(pred) == len(target)
    tp = 0
    for i in range(len(pred)):
        if pred[i] == target[i] and pred[i] == 1:
            tp += 1
    precision = tp / np.sum(pred == 1)
    recall = tp / np.sum(target == 1)
    f_score = (2 * precision * recall) / (precision + recall)
    return f_score, precision, recall


def round_average(stim_cfg, y_prob, evt_tar, round_num=None):
    """
    求每个字符（trial）的叠加平均，
    其中包括了 BLDA 预测结果 y_prob 和 event 和 target 等标签

    TODO:
    更灵活的 round 数量？

    input
    stim_cfg: StimulationConfig
    y_prob (stim_num * rounds * trials, 1) : 模型预测的 y 概率
    evt_tar (stim_num * rounds * trials, 2): 每个 trial 的 event(1 ~ 12) 和 target(0 or 1)
    num_round: round 的数量，若为 None, 则采用配置默认数量

    output
    y_prob_ave (stim_num * trials, 1): round 平均之后的 y_prob
    evt_tar_ave (stim_num * trials, 2): round 平均之后的 evt_tar
    """
    stim_num = stim_cfg.STIM_NUM
    if round_num is None:
        round_num = stim_cfg.ROUND_NUMBER
    stim_num_per_trial = stim_num * round_num  # 每个 trial 的刺激数量

    assert y_prob.shape[0] == evt_tar.shape[0]  # 确保长度相等
    assert y_prob.shape[0] % stim_num_per_trial == 0  # 确保余数为0
    trial_num = y_prob.shape[0] // stim_num_per_trial  # trial 的数量

    # 初始化输出变量
    y_prob_ave = np.zeros((trial_num * stim_num, 1), dtype=y_prob.dtype)
    evt_tar_ave = np.zeros((trial_num * stim_num, 2), dtype=evt_tar.dtype)

    for i_trial in range(trial_num):  # 循环取得每个 trial 的叠加平均
        start_ave, end_ave = i_trial * \
            stim_num, (i_trial + 1) * stim_num  # 取得每个 trial 的开始和结束的相对索引
        trial_bias = i_trial * stim_num_per_trial  # 每个 trial 对应的偏移量
        round_bias_list = np.arange(
            0, stim_num_per_trial, stim_num) + trial_bias  # 每个 round 的偏移量列表
        for round_bias in round_bias_list:  # 取得每个 trial 里每个 round 的偏移量（开始的索引）
            start, end = round_bias, round_bias + stim_num  # 每个 round 开始和结束的索引
            # 深拷贝方式取出对应 round 的数据
            curr_y_prob = y_prob[start:end, :].copy()
            curr_evt_tar = evt_tar[start:end, :].copy()

            # 获得按刺激编号从小到达排列的索引
            sorted_idx = np.argsort(curr_evt_tar[:, 0])
            # 重排 round 里的数据
            curr_y_prob = curr_y_prob[sorted_idx, :]
            curr_evt_tar = curr_evt_tar[sorted_idx, :]

            # 将结果放入对应输出变量中
            y_prob_ave[start_ave:end_ave, :] += curr_y_prob  # 注意 y_prob 是累加
            evt_tar_ave[start_ave:end_ave, :] = curr_evt_tar  # evt_tar 是覆盖
        y_prob_ave[start_ave:end_ave, :] /= round_num  # 对 y_prob 取均值

    return y_prob_ave, evt_tar_ave


def cal_char_acc(stim_cfg, y_prob, evt_tar, return_char=False):
    """
    计算字符准确率及对应预测字符

    尝试统一两个范式的字符定位方式，采用一下策略：
    1) 选出最大的 y probability 的 flash, 记其索引为 i,
    2) 根据 STIM_TARGET_RULE 获取 i 对应的若干字符的另一个 flash 的索引，形成列表 J (可行索引列表)
    3) 寻出 J 中索引对应的 trials 中 y probability 最大的 flash, 记其索引为 j
    4) 由 i, j 得到对应字符

    注1：若输入未经过 round averge, 这里的 trial 即为 trial 里的各个 round

    input
    stim_cfg: StimulationConfig
    y_prob (stim_num * trials, 1) : y 概率
    evt_tar (stim_num * trials, 2): event 和 target
    return_char: 是否返回 目标字符 与 预测字符

    output
    acc : 字符准确率
    [tar_char_list, pred_char_list]: 目标字符 与 预测字符 列表的列表
    """
    stim_num = stim_cfg.STIM_NUM
    stim_rule = stim_cfg.STIM_TARGET_RULE
    assert y_prob.shape[0] == evt_tar.shape[0]
    assert y_prob.shape[0] % stim_num == 0

    # 此处视数据为单 round，stim_num 个刺激被视为一个 trial
    trial_num = y_prob.shape[0] // stim_num

    res_bool = np.full(trial_num, False, dtype=bool)  # 初始化输出结果, 默认错误
    if return_char:
        tar_char_list, pred_char_list = [], []  # 初始化输出结果列表
    for i_trial in range(trial_num):
        start, end = i_trial * \
            stim_num, (i_trial + 1) * stim_num  # 每个 trail 开始和结束的索引

        # 深拷贝方式取出对应 trial 的数据
        curr_y_prob = y_prob[start:end, :].copy()
        curr_evt_tar = evt_tar[start:end, :].copy()

        # 获得按刺激编号从小到达排列的索引
        sorted_idx = np.argsort(curr_evt_tar[:, 0])
        # 重排：刺激顺序重排，输出的概率也根据此索引重排
        curr_y_prob = curr_y_prob[sorted_idx, :]
        curr_evt_tar = curr_evt_tar[sorted_idx, :]

        # 目标字符对应的刺激（闪烁）索引
        tar_flash_index = np.where(curr_evt_tar[:, 1])[0]

        if return_char:
            # 获取真实字符/标签
            # tar_char_index = find_char_from_flash(stim_rule, tar_flash_index[0], tar_flash_index[1])
            tar_char_index = np.where(curr_evt_tar[:, 1])[0]
            # 将目标字符添加到输出结果列表
            tar_char_list.append(stim_cfg.STIM_TARGET_TEXT[tar_char_index])

        # 获取预测字符对应的刺激的索引
        max_idx = np.argmax(curr_y_prob)  # 获取概率最大的 flash 索引
        # 找到第一个确定的 flash 可能匹配的索引
        # pm_idx, pm_char = find_possible_matched_idx(stim_rule, max_idx)  # pm -> possible matched
        # 获取可行的概率第二大 flash 索引
        # sub_max_idx = pm_idx[np.argmax(curr_y_prob[pm_idx, :])]
        # 判断
        # if (max_idx in tar_flash_index) and (sub_max_idx in tar_flash_index):
        if max_idx == tar_flash_index:
            res_bool[i_trial] = True  # 第一个和第二个 flash 索引均正确, 则预测正确

        if return_char:
            # 获取预测字符索引及对应字符
            pred_char_index = max_idx
            # 将预测字符添加到输出结果列表
            pred_char_list.append(stim_cfg.STIM_TARGET_TEXT[pred_char_index])

    acc = np.sum(res_bool) / trial_num  # 计算平均准确率

    if return_char:
        assert len(tar_char_list) == len(pred_char_list)
        return acc, [tar_char_list, pred_char_list]
    else:
        return acc


def find_element(flash_to_char, char):
    """
    根据刺激规则（flash_to_char=StimulationConfig.STIM_TARGET_RULE）
    返回与 char对应的 flash 索引

    input
    flash_to_char: StimulationConfig.STIM_TARGET_RULE
    char: 带查找 flash 索引的字符

    output
    res: 对应 flash 索引的列表
    """
    res = []
    for i_flash, chars in enumerate(flash_to_char):
        if char in chars:
            res.append(i_flash)
    return res  # list


def find_char_from_flash(flash_to_char, i, j):
    """
    根据刺激规则（flash_to_char=StimulationConfig.STIM_TARGET_RULE）
    返回第 j 闪与第 i 闪对应的字符索引

    注1：根据 stim rule 的特性，任意两个 flash 对应的字符列表的交集只可能是 1 个 char 或 0 个 char

    input
    flash_to_char: StimulationConfig.STIM_TARGET_RULE
    i: 待查找字符的第一个 flash 索引
    j: 待查找字符的第二个 flash 索引

    output
    返回查找到的字符索引
    """
    # 取交集
    intersection = list(
        set(flash_to_char[i]).intersection(set(flash_to_char[j])))
    return intersection[0] if len(intersection) == 1 else None


def find_possible_matched_idx(flash_to_char, i):
    """
    根据刺激规则（flash_to_char=StimulationConfig.STIM_TARGET_RULE）
    返回可能与第 i 闪配对的闪烁的索引列表 和 配对后相应字符索引列表

    注1：根据 stim rule 的特性，任意两个 flash 对应的字符列表的交集只可能是 1 个 char 或 0 个 char
    注2：TODO 从图论的角度或许有更优的解法
    注3：TODO 设置缓存也可以将后期复杂度降到 O(1).

    input
    flash_to_char: StimulationConfig.STIM_TARGET_RULE
    i: 待匹配的 flash 索引

    output
    pm_idx: 可能与第 i 闪配对的闪烁的索引列表 pm -> possible matched
    pm_char: 配对后相应字符索引列表
    """
    assert isinstance(flash_to_char[0],
                      list), "first arg should be list of list."
    pm_idx = []  # 初始化可能匹配 flash索引 列表
    pm_char = []  # 初始化可能匹配 字符索引 列表
    i_char_set = set(flash_to_char[i])  # 取出第 i 闪字符索引备用
    for j_flash, j_chars in enumerate(flash_to_char):
        # 用第 i 行对非第 i 行里的索引取交集
        if j_flash != i:
            intersection = list(i_char_set.intersection(set(j_chars)))
            if len(intersection) > 0:  # 若取得交集非空
                pm_idx.append(j_flash)  # 添加对应 flash 索引
                pm_char.append(intersection[0])  # 添加对应字符索引
    # 找到的可能匹配 flash 应该数量等于第 i 闪的字符数
    assert len(pm_idx) == len(flash_to_char[i])
    return pm_idx, pm_char  # list, list


def z_score(data):
    """
    将特征在时间序列（time_series）维度上进行标准化

    input
    data (channels, time_series, stims): 原始特征

    output
    Y (channels, time_series, stims): 标准化后的 data

    """
    n_trl = data.shape[2]
    Y = np.zeros_like(data)
    for i in range(n_trl):
        # 深拷贝，x成为 Y 的引用， 保持 data 不被修改
        Y[:, :, i] = data[:, :, i]
        x = Y[:, :, i]

        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        # 需要处理方差为零的情况
        nz_idx = ~np.squeeze(x_std == 0)
        x[~nz_idx, :] = 0
        # 处理方差不为 0 的情况
        x[nz_idx, :] = (x[nz_idx, :] - x_mean[nz_idx]) / x_std[nz_idx]
    return Y
