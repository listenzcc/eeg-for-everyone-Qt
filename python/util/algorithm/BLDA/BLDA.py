
import numpy as np
from .Supplementary_function import z_score
import pickle
import os


def post_process_y_prob(y_prob):
    """
    后处理，此处只实现了到 [0, 1] 的映射

    input
    y_prob (stims, 1): 概率 y

    output
    y_prob_n (stims, 1): 后处理后的 y_prob
    """
    y_prob_n = (y_prob - np.min(y_prob)) / \
        (np.max(y_prob) - np.min(y_prob))  # 映射到 (0, 1)
    y_prob_n[y_prob_n > 0.5] = 1  # 大于 0.5 取 1
    y_prob_n[y_prob_n < 0.5] = 0  # 小于 0.5 取 0
    return y_prob_n


class BLDA:
    '''BLDA method from Bei Wang'''

    def __init__(self, name, cfg=None):
        # 为每一个 BLDA 设置一个独一无二的 name
        self.name = name  # 这里指模型的命名
        self.W = None                   # 权重
        self.is_trained = False         # 是否训练好了，初始时都为未训练

        # 参数
        if cfg is not None:  # TODO 关于 BLDA 的参数设置
            NotImplementedError("[Customized]")
        else:  # 目前参数写固定
            self.cfg = {
                # (initial) inverse variance of prior distribution
                'alpha': 25,
                # (initial) inverse variance of prior for bias term
                'biasalpha': 0.00000001,
                'beta': 1,  # (initial) inverse variance around targets
                'stopeps': 0.0001,  # desired precision for alpha and beta
                'maxit': 500,  # maximal number of iterations
            }

    def load(self, model, verbose=True):
        """
        加载模型，完成对实例中关键属性的修改（如，加载训练好的权重）

        input
        model: 可以接受两种加载模式，模型路径 或 模型字典
        verbose: 是否打印一些信息
        """
        if isinstance(model, str):
            with open(model, 'rb') as f:
                model = pickle.load(f)
        assert isinstance(model, dict)

        # name 不做替换
        self.W = model['W']
        self.cfg = model['cfg']
        self.is_trained = model['is_trained']

        if verbose:
            print(f"Loading <{model['name']}> to <{self.name}> is done.")

    def save(self, save_path=None, verbose=True):
        """
        保存模型，字典形式返回模型关键部分
        通过 pickle 序列化保存到硬盘指定位置

        inputs
        save_path: 若为 None, 则不保存在硬盘; 否则据此参数保存到对应位置
        verbose: 是否打印一些信息

        outputs
        model (dict): 模型中关键部分
        """
        assert self.is_trained, "模型还未被训练，请先训练后再保存"
        # 这里需要把模型中最关键的部分保存下来
        model = {
            'name': self.name,
            'W': self.W,
            'is_trained': self.is_trained,
            'cfg': self.cfg
        }
        if save_path is not None:
            save_file = os.path.join(save_path, f"{self.name}.pkl")
            with open(save_file, 'wb') as f:
                pickle.dump(model, f)
            if verbose:
                print(f"Model {self.name} have been saved at {save_file}.\n")

        return model

    # TODO 请确认 BLDA 算法的正确性 ！！！
    def fit(self, train_x, ylabel):
        """
        模型训练接口， 利用 特征向量 和 标签 进行训练

        inputs
        train_x's shape     : (channels, time_series, stims)  stims 一轮12闪
        ylabel's shape      : (stims, 1)

        ? --------------------
        ? Question from Chuncheng Zhang
        ? 
        ? 23 channels are required
        ? The data's w size is (3151, 1)
        ? 3151 / 23 = 137
        ! The question is how does the 137 match with the time series
        ? --------------------

        BLDA status
        W  shape            : (channels * time_series, 1)
        is_train            : True

        注1： 深浅拷贝的问题
        """
        x = z_score(train_x)
        y = np.ones_like(ylabel)
        y[ylabel == 0] = -1
        # 以上保证了对 x, y 的操作不对原始数据产生影响

        # 数据相关常量
        totalNum = x.shape[2]
        n_posexamples = list(y).count(1)
        n_negexamples = list(y).count(-1)
        n_examples = n_posexamples + n_negexamples

        # 展平数据 (channels, time_series, stims) -> (channels * time_series, stims)
        x = np.reshape(x, (x.shape[0] * x.shape[1], totalNum)).astype('float')
        # first_=True
        # for ii in range (x.shape[2]):
        #     if first_:
        #         first_=False
        #         temp_=x[:,:,ii]
        #     else:
        #         temp_=np.concatenate((temp_,x[:,:,ii]),axis=1)
        #
        # pass
        y = (y.T).astype('float')

        # 调整标签敏感度，注意应该为浮点除法
        y[y == 1] = n_examples / float(n_posexamples)
        y[y == -1] = -n_examples / float(n_negexamples)
        # 未标签增加 bias
        x = np.vstack((x, np.ones((1, x.shape[1]))))

        # 算法迭代初始值和常量
        n_features = x.shape[0]  # dimension of feature vectors
        d_beta = np.inf  # (initial) diff. between new and old beta
        d_alpha = np.inf  # (initial) diff. between new and old alpha

        # 可调参数
        # (initial) inverse variance of prior distribution
        alpha = self.cfg['alpha']
        # (initial) inverse variance of prior for bias term
        biasalpha = self.cfg['biasalpha']
        beta = self.cfg['beta']  # (initial) inverse variance around targets
        stopeps = self.cfg['stopeps']  # desired precision for alpha and beta
        maxit = self.cfg['maxit']  # maximal number of iterations

        i = 1  # keeps track of number of iterations

        # 协方差阵维度：(channels * time_series)^2
        valuee, vectore = np.linalg.eigh(x.dot(x.T))  # eigh 针对对称阵，返回实特征根
        sortedIndex = np.argsort(valuee)
        valuee = valuee[sortedIndex]
        vectore = vectore[:, sortedIndex]
        vxy = vectore.T.dot(x).dot(y.T)
        d = valuee.reshape(valuee.shape[0], 1)
        e = np.ones((n_features - 1, 1)).astype('double')
        v = vectore
        while ((d_alpha > stopeps) or (d_beta > stopeps)) and (i < maxit):
            alphaold = alpha
            betaold = beta
            ss = np.vstack((alpha * e, biasalpha))
            m = (beta * v).dot(1. / (beta * d + ss) * vxy)
            err = sum(np.square(y - m.T.dot(x)).flatten())
            gamma = sum((beta * d / (beta * d + ss)).flatten())
            alpha = gamma / (m.T.dot(m))
            beta = (n_examples - gamma) / err
            d_alpha = np.abs(alpha - alphaold)
            d_beta = np.abs(beta - betaold)

            i += 1

        self.W = m  # 将权重记录在类属性 W 中
        self.is_trained = True  # 训练好了标准设置为 True

    def predict(self, test_x):
        """
        # 模型测试接口， 输入特征向量输出预测概率

        inputs
        test_x's shape     : (channels, time_series, stims)

        outputs
        y_prob's shape      : (stims, 1)
        """
        assert self.is_trained, "模型还未被训练, 请先训练后再测试"

        x = z_score(test_x)
        # 以上保证了对 x 的操作不对原始数据产生影响

        numT = x.shape
        totalNum = numT[2]

        # 展平数据 (channels, time_series, stims) -> (channels * time_series, stims)
        x = np.reshape(x, (numT[0] * numT[1], totalNum)).astype('float')
        x = np.vstack((x, np.ones((1, x.shape[1]))))  # 增加全 1 偏置对齐特征维度
        # (1, channels * time_series + 1) * (channels * time_series + 1, stims)
        y_prob = self.W.T.dot(x)

        return y_prob.T  # (stims, 1)
