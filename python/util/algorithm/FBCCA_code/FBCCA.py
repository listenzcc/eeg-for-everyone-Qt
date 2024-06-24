import numpy as np


class FBCCA:
    def __init__(self, target_template_set, a, b, nFB):
        # 正余弦参考信号
        self.target_template_set = target_template_set
        self.a = a
        self.b = b
        self.nFB = nFB
        self.weights = pow(np.linspace(1, self.nFB, self.nFB), self.a) + self.b

    def predict(self, dataFB):
        # The data shape is (nFB, channels, timePoints)
        p = np.zeros([len(self.target_template_set), self.nFB])
        # print(weights)
        for i in range(0, len(dataFB)):
            data = dataFB[i]
            data = data.T
            # qr分解,data:length*channel
            [Q_temp, R_temp] = np.linalg.qr(data)
            for j in range(0, len(self.target_template_set)):
                template = self.target_template_set[j]
                template = template[:, 0:data.shape[0]]
                template = template.T
                [Q_cs, R_cs] = np.linalg.qr(template)
                data_svd = np.dot(Q_temp.T, Q_cs)
                [U, S, V] = np.linalg.svd(data_svd)
                rho = pow(S[0], 2)*self.weights[i]
                p[j, i] = rho
        pFB = np.sum(p, axis=1)
        return pFB.argmax()

    def fit(self, dataFB, template):
        A = []
        B = []
        rho = []
        template = template[:, 0:dataFB[0].shape[1]]
        template = template.T
        # print(weights)
        for i in range(0, self.nFB):
            data = dataFB[i]
            data = data.T
            # qr分解,data:length*channel
            [Q_temp, R_temp] = np.linalg.qr(data)
            [Q_cs, R_cs] = np.linalg.qr(template)
            data_svd = np.dot(Q_temp.T, Q_cs)
            [U, S, V] = np.linalg.svd(data_svd)
            A.append(np.dot(np.linalg.inv(R_temp), U))
            B.append(np.dot(np.linalg.inv(R_cs), V.T))
            rho.append(S)

        return A, B, rho
