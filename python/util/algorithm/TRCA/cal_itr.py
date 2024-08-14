import math
'''
计算ITR
    输入：
        - T：单词计算使用数据时长（单位：秒），理论ITR直接使用数据长度对应时间，实际ITR需要考虑视线转移时间等因素，例如T =（数据长度+0.5）
        * 没做特别要求则可以将视线转移时间默认设定为0.5秒
        - N：目标数量，例如40目标系统则N = 40
        - P：准确率，这里当P<1/N，即小于机会水平时，ITR直接返回0
    输出：
        - ITR：信息传输速率（单位：bits/min）
'''
def cal_itr(T,N,P):
    if P < 1/N:
        ITR = 0
    elif P == 1:
        ITR = (1 / T) * (math.log(N, 2)) * 60
    else:
        ITR = (1 / T) * (math.log(N, 2) + (1 - P) * math.log((1 - P) / (N - 1), 2) + P * math.log(P, 2)) * 60
    return ITR
    # print(ITR)
