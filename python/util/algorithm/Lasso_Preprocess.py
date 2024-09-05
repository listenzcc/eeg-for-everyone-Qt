from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class Lasso_Process:
    """
    input
        data:(stems,channels,length)
        event:(stems,1) 0 or 1

    output
        select channel's index
    """

    def __init__(self, **kwargs):
        self.data = kwargs.get('Data', None)
        self.events = kwargs.get('Events', None)
        self.montage = kwargs.get('montage',
                                  ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3',
                                   'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
                                   'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1',
                                   'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6',
                                   'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2',
                                   'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'])

        self.select_channels = None
        self.channel_importance = kwargs.get("importance", None)
        self.alpha = kwargs.get("alpha", 0.01)
        self.select_data = None
        self.test_size = kwargs.get("test_size", 0.3)

    def fit(self):
        if self.data is None or not self.data.any():
            raise ValueError("数据未正常加载！")
        if self.events is None or not self.events.any():
            raise ValueError("事件未正常加载！")

        X = self.data.reshape(self.data.shape[0], -1)
        y = self.events

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)
        lasso = Lasso(self.alpha)
        lasso.fit(X_train, y_train)

        lasso_coefficients = lasso.coef_.reshape(
            self.data.shape[1], self.data.shape[2])
        self.channel_importance = np.sum(np.abs(lasso_coefficients), axis=1)

        self.select_channels = np.where(self.channel_importance)[0]
        self.select_data = self.data[:, self.select_channels, :]

        return self.select_channels
