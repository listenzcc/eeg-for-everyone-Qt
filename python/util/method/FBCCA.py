import numpy as np


class FBCCA:
    def __init__(self, target_template_set):
        # 正余弦参考信号
        self.target_template_set = target_template_set

    def recognize(self, dataFB):
        fb_number = len(dataFB)
        p = np.zeros([len(self.target_template_set), fb_number])
        weights = pow(np.linspace(1, fb_number, fb_number), -1.25) + 0.25
        # print(weights)
        for i in range(0,len(dataFB)):
            data = dataFB[i]
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
                rho = pow(S[0], 2)*weights[i]
                p[j,i] = rho
        pFB = np.dot(p , weights)
        result = np.where(pFB==np.max(pFB))
        result = result[0]+1
        return result[0]
