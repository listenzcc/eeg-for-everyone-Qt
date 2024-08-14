import numpy as np
# 数据维度：trials*channels*timepoint


class TRCA:
    '''
    TRCA算法
    模型初始化：
        trca_method = TRCA(Type_num)
        Type_num (int)为类别数量，例如40目标系统则Type_num = 40

        TRCA的方法：
            - trca_method.fit(Data_train, Type_train)，模型训练
            - trca_method.predict_proba(model, Data_test)，模型测试。输出score为样本对应的各个类别的评分
            - trca_method.predict(model, Data_test)，模型测试，输出type_test为样本对应的类别

        输入参数：
            - Data_train (np.ndarray)，训练数据，维度为 (trials_train,channels,timepoint)
            - Type_train (np.ndarray)，训练数据标签，维度为(trials_train,)，trials_train与Data_train维度一致，
            * Type_train中每一类应至少有1个trial，即1到Type_num所有数字应至少出现一次。
            - Data_test (np.ndarray)，测试数据，维度为 (trials_test,channels,timepoint)，其中channels和timepoint应与Data_train一致
            - model (dict)，数据经过trca_method.fit得到的训练模型，包括：
                - T (np.ndarray)，参考脑电模板，维度为(Type_num, channels, timepoint)，channels和timepoint与Data_train一致
                - W (np.ndarray)，空间滤波器，维度为(Type_num, channels)，channels与Data_train一致

        输出参数：
            - model (dict)，数据经过trca_method.fit得到的训练模型，包括：
                - T (np.ndarray)，参考脑电模板，维度为(Type_num, channels, timepoint)，channels和timepoint与Data_train一致
                - W (np.ndarray)，空间滤波器，维度为(Type_num, channels)，channels与Data_train一致
            - score (np.ndarray)，数据经过trca_method.predict_proba得到的预测得分，维度为(trials_test,Type_num)，trials_test与Data_test一致
            - type_test (np.ndarray)，数据经过trca_method.predict得到的预测结果，维度为(trials_test,)，trials_test与Data_test一致
    '''

    def __init__(self, Type_num):
        # 标签从1开始计数
        self.Type_num = Type_num

    def predict(self, Data_test, model=None):
        if not model:
            model = self.model

        # nChans = np.size(Data_test, 1)
        # nPoints = np.size(Data_test, 2)
        data_test = np.transpose(Data_test, axes=(1, 2, 0))
        W = model['W']
        T = model['T']
        type_test = np.zeros(np.size(Data_test, 0))
        for Ntest in range(np.size(Data_test, 0)):

            data_trial = data_test[:, :, Ntest]
            p = []
            # predict
            for TypeIndex in range(self.Type_num):
                # DA=np.dot(W[frequencyIndex,:,:],data_trial)
                # DB=np.dot(W[frequencyIndex,:,:],T[frequencyIndex,:,:])

                DA = np.dot(W, data_trial)
                DB = np.dot(W, T[TypeIndex, :, :])
                DA = DA - np.mean(DA)
                DB = DB - np.mean(DB)
                U = np.sum(np.multiply(DA, DB))
                D = np.sqrt(np.sum(np.multiply(DA, DA))
                            * np.sum(np.multiply(DB, DB)))
                rho = U / D
                p.append(rho)
            result = p.index(max(p))
            type_test[Ntest] = round(result)+1

        return type_test

    def predict_proba(self, Data_test, model=None):
        if not model:
            model = self.model

        # nChans = np.size(Data_test, 1)
        # nPoints = np.size(Data_test, 2)
        data_test = np.transpose(Data_test, axes=(1, 2, 0))
        W = model['W']
        T = model['T']
        score = np.zeros([np.size(Data_test, 0), self.Type_num])
        for Ntest in range(np.size(Data_test, 0)):

            data_trial = data_test[:, :, Ntest]
            p = []
            # predict
            for TypeIndex in range(self.Type_num):
                # DA=np.dot(W[frequencyIndex,:,:],data_trial)
                # DB=np.dot(W[frequencyIndex,:,:],T[frequencyIndex,:,:])

                DA = np.dot(W, data_trial)
                DB = np.dot(W, T[TypeIndex, :, :])
                DA = DA - np.mean(DA)
                DB = DB - np.mean(DB)
                U = np.sum(np.multiply(DA, DB))
                D = np.sqrt(np.sum(np.multiply(DA, DA))
                            * np.sum(np.multiply(DB, DB)))
                rho = U / D
                score[Ntest, TypeIndex] = rho

        # Convert positive infinity to 1, negative infinity to 0 using the sigmoid function
        proba = 1 / (1 + np.exp(-score))

        # Normalize the rows to sum=1
        for e in proba:
            e /= np.sum(e)

        return proba

    def fit(self, Data_train, Type_train):
        '''
        Train the totally new model.

        Args:
            - Data_train: Training X with (stimulus, channels, time points)
            - Type_train: Training y with (stimulus,)
        '''

        nChans = np.size(Data_train, 1)
        nPoints = np.size(Data_train, 2)
        T = np.zeros([self.Type_num, nChans, nPoints])
        temp = np.zeros([nChans, 1])
        S = np.zeros([self.Type_num, nChans, nChans])
        for ii in range(self.Type_num):
            index_1 = np.where(Type_train == ii+1)

            data_1 = Data_train[index_1[0], :, :].mean(axis=0)
            temp[:, 0] = data_1.mean(axis=1)
            T[ii, :, :] = data_1 - temp

            SS = Data_train[index_1[0], :, :]
            SS = SS.sum(axis=0)
            SS = (SS.T - SS.mean(axis=1)).T
            S[ii, :, :] = np.dot(SS, SS.T)

        Q = np.transpose(Data_train, axes=(1, 2, 0)).reshape((nChans, -1))
        Q = (Q.T - Q.mean(axis=1)).T
        Q = np.dot(Q, Q.T)
        Q1 = np.linalg.pinv(Q)

        W = np.zeros([self.Type_num, nChans])
        for ii in range(self.Type_num):
            D, Wt = np.linalg.eig(np.dot(Q1, S[ii, :, :]))
            W[ii, :] = Wt[:, np.argmax(D)]

        model = dict(
            T=T,
            W=W
        )

        self.model = model

        return model
