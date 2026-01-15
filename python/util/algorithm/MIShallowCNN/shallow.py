import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def shallow_cnn_decoding(epochs, events, n_folds=5, epochs_train=50, batch_size=32):
    """
    ShallowCNN解码方法实现

    参数:
        epochs: mne.Epochs对象
        events: 事件标签数组
        n_folds: 交叉验证折数 (默认5)
        epochs_train: 训练轮数 (默认50)
        batch_size: 批大小 (默认32)

    返回:
        mean_accuracy: 平均准确率
        all_accuracies: 每折的准确率
        fig: 混淆矩阵的matplotlib Figure对象
    """
    X = epochs.get_data()  # 形状 (n_epochs, n_channels, n_times)
    y = events[:, -1]  # 获取事件标签
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # 验证至少有2类
    if n_classes < 2:
        raise ValueError(f"需要至少2类，但只有{n_classes}类")

    # 数据预处理：增加通道维度 (用于CNN输入)
    X = X[..., np.newaxis]  # 形状变为 (n_epochs, n_channels, n_times, 1)

    # 初始化交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_accuracies = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 验证训练集和测试集都包含所有类
        for split_name, split_X, split_y in [
            ('训练集', X_train, y_train),
            ('测试集', X_test, y_test)
        ]:
            if len(np.unique(split_y)) < 2:
                raise ValueError(
                    f"第{fold_idx}折{split_name}只包含{len(np.unique(split_y))}类。"
                    "请检查数据分割或增加样本量。"
                )

        # 转换为分类需要的one-hot编码
        y_train_cat = to_categorical(y_train - min(y), num_classes=n_classes)
        y_test_cat = to_categorical(y_test - min(y), num_classes=n_classes)

        # 创建ShallowCNN模型
        model = build_shallow_cnn(
            input_shape=(X.shape[1], X.shape[2], X.shape[3]),
            n_classes=n_classes
        )

        # 训练模型
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs_train,
            batch_size=batch_size,
            verbose=1
        )

        # 预测并计算准确率
        y_pred = model.predict(X_test).argmax(axis=1) + min(y)
        acc = np.mean(y_pred == y_test)
        all_accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold_idx} accuracy: {acc:.3f}")
        print(
            f"Fold {fold_idx} training history: val_acc max {max(history.history['val_accuracy']):.3f}")

    # 计算平均准确率
    mean_accuracy = np.mean(all_accuracies)
    print(f"\nMean accuracy: {mean_accuracy:.3f}")

    # 创建混淆矩阵图形
    cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f'Class {int(c)}' for c in unique_classes]
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(
        f'ShallowCNN Confusion Matrix (Mean Acc: {mean_accuracy:.3f})')
    plt.tight_layout()

    return mean_accuracy, all_accuracies, fig


def build_shallow_cnn(input_shape, n_classes):
    """构建ShallowCNN模型架构"""
    model = Sequential([
        # 第一卷积层
        Conv2D(40, (1, 25), activation='elu',
               input_shape=input_shape,
               padding='same'),
        # 第二卷积层
        Conv2D(40, (input_shape[0], 1),
               use_bias=False, activation='linear',
               padding='valid'),
        BatchNormalization(axis=1),
        # 非线性激活
        tf.keras.layers.Activation('elu'),
        Dropout(0.5),
        # 分类层
        Flatten(),
        Dense(n_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
