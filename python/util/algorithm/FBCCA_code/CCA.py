import numpy as np
# from scipy.stats import pearsonr

class CCA:
    def __init__(self, target_template_set):
        # 正余弦参考信号
        self.target_template_set = target_template_set

    def predict(self, data):
        p = np.zeros([len(self.target_template_set), 1])
        data = data.T
        # qr分解,data:length*channel
        [Q_temp, R_temp] = np.linalg.qr(data)
        for j in range(0,len(self.target_template_set)):
            template = self.target_template_set[j]
            template = template[:, 0:data.shape[0]]
            template = template.T
            [Q_cs, R_cs] = np.linalg.qr(template)
            data_svd = np.dot(Q_temp.T, Q_cs)
            [U, S, V] = np.linalg.svd(data_svd)
            rho = S[0]
            p[j] = rho

        return p.argmax()

    def fit(self, data, template):
        data = data.T
        # qr分解,data:length*channel
        [Q_temp, R_temp] = np.linalg.qr(data)
        template = template[:, 0:data.shape[0]]
        template = template.T
        [Q_cs, R_cs] = np.linalg.qr(template)
        data_svd = np.dot(Q_temp.T, Q_cs)
        [U, S, V] = np.linalg.svd(data_svd)

        A = np.dot(np.linalg.inv(R_temp), U)
        B = np.dot(np.linalg.inv(R_cs), V.T)
        rho = S
        # A和B为空间滤波器，np.dot(data,A[:,i])和np.dot(template,B[:,i])的相关系数等于rho[i]，即：
        # c = pearsonr(np.dot(data.T, CCA_A[:, i]), np.dot(template, CCA_B[:, i]))
        # c[0]与rho[i]相等
        return A, B, rho