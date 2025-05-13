import numpy as np


def mutual_information_from_confusion_matrix(confusion_matrix):
    """
    计算混淆矩阵的互信息 I(Y; Y_hat)

    参数:
        confusion_matrix (np.ndarray或list): k x k 的混淆矩阵

    返回:
        float: 互信息 I(Y; Y_hat)（单位：比特）
    """
    # 转换为NumPy数组并确保是二维的
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    assert cm.ndim == 2, "混淆矩阵必须是二维的"
    assert cm.shape[0] == cm.shape[1], "混淆矩阵必须是方阵"

    # 计算总样本数
    N = np.sum(cm)
    if N == 0:
        return 0.0  # 避免除以零

    # 计算联合概率 P(Y, Y_hat)
    joint_prob = cm / N

    # 计算边际概率 P(Y) 和 P(Y_hat)
    p_true = np.sum(joint_prob, axis=1)  # P(Y)
    p_pred = np.sum(joint_prob, axis=0)  # P(Y_hat)

    # 计算熵 H(Y)
    h_true = -np.sum([p * np.log2(p) for p in p_true if p > 0])

    # 计算条件熵 H(Y | Y_hat)
    h_cond = 0.0
    for j in range(cm.shape[1]):
        p_j = p_pred[j]
        if p_j == 0:
            continue
        conditional_p = joint_prob[:, j] / p_j  # P(Y | Y_hat=j)
        h_cond_j = -np.sum([p * np.log2(p) for p in conditional_p if p > 0])
        h_cond += p_j * h_cond_j

    # 互信息 I(Y; Y_hat) = H(Y) - H(Y | Y_hat)
    mi = h_true - h_cond
    return mi


# 示例用法
if __name__ == "__main__":
    # 示例混淆矩阵（3x3）
    confusion_matrix = [
        [50, 10, 5],
        [5, 20, 0],
        [0, 5, 5]
    ]
    confusion_matrix = [
        [50, 0, 0, 0],
        [0, 50, 0, 0],
        [0, 0, 50, 0],
        [0, 0, 0, 50],
    ]

    mi = mutual_information_from_confusion_matrix(confusion_matrix)
    print(f"互信息 I(Y; Y_hat) = {mi:.4f} 比特")
