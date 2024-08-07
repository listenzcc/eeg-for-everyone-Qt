import warnings
from pathlib import Path

from .EEGNetFramework import EEGNetFramework
import torch.utils.data as Data
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


class EEGNet(object):

    def __init__(self, **kwargs):
        self.freq = kwargs.get('DOWN_SAMPLE', 250)
        self.channel_num = kwargs.get('CHANNEL_NUM', 39)
        self.time_point_num = kwargs.get('TIME_POINT_NUM', 200)
        self.sample_w = kwargs.get('SAMPLE_WEIGHT', [1.0, 8.0])
        self.train_batch_size = kwargs.get('TRAIN_BATCH_SIZE', 512)
        self.val_batch_size = kwargs.get('VAL_BATCH_SIZE', 512)
        self.test_batch_size = kwargs.get('TEST_BATCH_SIZE', 512)
        self.epoch_n = kwargs.get('EPOCH_N', 200)
        self.lr = kwargs.get('LR', 0.001)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = kwargs.get('MODEL_PATH', None)
        self.trained = False  # 标识是否已经加载或训练模型
        self.net = EEGNetFramework(
            self.freq, self.channel_num, self.time_point_num).to(self.device)

        # Cuda not available warning
        if self.device == 'cpu':
            warnings.warn('The cuda is not available, using cpu instead.')

        # If the model_path is not available, using the default.pkl besides the py file
        if not self.model_path:
            self.model_path = Path(__file__).parent.joinpath('default.pkl')
            warnings.warn(
                f'The model_path is not provided, using default: {self.model_path}')

        # Load the model parameters from the model_path
        if self.model_path:
            self.__load_model(self.model_path)

    def __load_model(self, model_path):
        with open(model_path, 'rb') as file:
            state_dict = torch.load(file, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.trained = True

    def fit(self, x, y):
        if self.trained:
            print("Model already loaded or trained, skipping training.")
            return
        """
        input:
            x（Stims , Seleted_Channel_Nums, Length):（刺激数，选取通道数目，数据长度）
            y:一维数组,[Stims_label] ; 1 or 0
            
        output:
            net
        """
        train_data = np.expand_dims(x, axis=1)
        train_evt = np.eye(2, dtype=np.int32)[y.squeeze().astype(np.int32)]
        train_dataset = Data.TensorDataset(torch.Tensor(
            train_data), torch.FloatTensor(train_evt))
        train_loader = Data.DataLoader(
            dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.sample_w).to(self.device))
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in range(self.epoch_n):
            self.net.train()
            train_bar = tqdm(train_loader)  # 把 dataloader 放到一个 进度条管理器（tqdm） 里
            train_bar.set_description('Epoch %d' % (epoch + 1))
            for train_x, train_y in train_bar:
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                logits, probas = self.net(train_x)
                loss = criterion(logits, train_y[:, 1].long())

                train_bar.set_postfix(
                    {'train loss': '{0:1.5f}'.format(loss.data.item()), })
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.trained = True  # 标记模型已训练
        self.net.eval()

    def predict_proba(self, x):
        if not self.trained:
            raise ValueError(
                "Model is not trained. Please train the model or load a pretrained model.")
        """
        input:
            x（Stims , Seleted_Channel_Nums, Length):（刺激数，选取通道数目，数据长度）

        output:
            y_prob
        """
        test_data = np.expand_dims(x, axis=1)
        fake_test_evt = np.eye(2, dtype=np.int32)[np.ones(
            test_data.shape[0]).astype(np.int32)]  # get one hot
        net = self.net.to(self.device)
        test_dataset = Data.TensorDataset(torch.Tensor(
            test_data), torch.FloatTensor(fake_test_evt))
        test_loader = Data.DataLoader(
            dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False)

        net.eval()
        res_all = []
        with torch.no_grad():  # 使得模型推理的时候不带梯度
            test_bar = tqdm(test_loader)  # 把 dataloader 放到一个进度条管理器（tqdm） 里
            test_bar.set_description('Epoch %d' % 1)
            for test_x, test_y in test_bar:
                test_x = test_x.to(self.device)
                _, probas = net(test_x)
                res_all.append(probas.cpu().numpy())

        # (sample_number, 2), 0: non target, 1: target
        output_res = np.concatenate(res_all, axis=0)
        y_prob = output_res[:, 1]

        return y_prob

    def predict(self, x):
        if not self.trained:
            raise ValueError(
                "Model is not trained. Please train the model or load a pretrained model.")
        """
        input:
            x（Stims , Seleted_Channel_Nums, Length):（刺激数，选取通道数目，数据长度）

        output:
            y_pred
        """
        test_data = np.expand_dims(x, axis=1)
        fake_test_evt = np.eye(2, dtype=np.int32)[np.ones(
            test_data.shape[0]).astype(np.int32)]  # get one hot
        net = self.net.to(self.device)
        test_dataset = Data.TensorDataset(torch.Tensor(
            test_data), torch.FloatTensor(fake_test_evt))
        test_loader = Data.DataLoader(
            dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False)

        net.eval()
        res_all = []
        with torch.no_grad():  # 使得模型推理的时候不带梯度
            test_bar = tqdm(test_loader)  # 把 dataloader 放到一个进度条管理器（tqdm） 里
            test_bar.set_description('Epoch %d' % 1)
            for test_x, test_y in test_bar:
                test_x = test_x.to(self.device)
                # test_y = test_y.to(self.device)
                _, probas = net(test_x)
                res_all.append(probas.cpu().numpy())

        # (sample_number, 2), 0: non target, 1: target
        output_res = np.concatenate(res_all, axis=0)
        y_pred = np.argmax(output_res, axis=1).reshape(-1, 1)
        return y_pred.squeeze()
