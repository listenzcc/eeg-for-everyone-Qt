import io
import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nicegui import ui
from pathlib import Path
from scipy import signal
from typing import List, Dict, Optional


class EEGAnalyzer:
    def __init__(self):
        self.window_size = 500  # 窗口大小
        self.step_size = 250  # 滑动步长
        self.Fs = 500  # 采样率Hz
        self.Normalization = True  # 是否归一化
        self.current_file = None
        self.df = None
        self.channels = None
        self.data = None
        self.num_channels = 0
        self.band_powers = None
        self.band_names = [
            'Delta (0.5-4Hz)', 'Theta (4-7Hz)', 'Alpha (8-13Hz)', 'Beta (14-25Hz)']

    def load_data(self, file_path: str):
        """加载EEG数据文件"""
        try:
            print(file_path)
            self.current_file = Path(file_path.name).name
            # 读取数据，跳过前两列（假设前两列是时间或其他非EEG数据）
            self.df = pd.read_csv(file_path)
            self.channels = self.df.columns[2:]
            self.data = self.df[self.channels].to_numpy()
            print(self.data.shape, self.df, self.channels)
            self.num_channels = self.data.shape[1] if len(
                self.data.shape) > 1 else 1
            if self.num_channels == 1:
                self.data = self.data.reshape(-1, 1)
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            ui.notify(f"加载文件失败: {str(e)}", type='negative')
            return False

    def compute_band_powers(self):
        """计算各频带能量"""
        if self.data is None:
            return False

        num_windows = (self.data.shape[0] -
                       self.window_size) // self.step_size + 1
        self.band_powers = np.zeros(
            (num_windows, self.num_channels, 4))  # 4个频带

        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            window_data = self.data[start:end, :]

            for ch in range(self.num_channels):
                # 计算功率谱密度
                f, Pxx = signal.welch(
                    window_data[:, ch], fs=self.Fs, nperseg=min(256, self.window_size))

                # 计算各频带能量
                delta_idx = np.logical_and(f >= 0.5, f < 4)
                theta_idx = np.logical_and(f >= 4, f < 7)
                alpha_idx = np.logical_and(f >= 8, f < 13)
                beta_idx = np.logical_and(f >= 14, f < 25)

                self.band_powers[i, ch, 0] = np.sum(Pxx[delta_idx])
                self.band_powers[i, ch, 1] = np.sum(Pxx[theta_idx])
                self.band_powers[i, ch, 2] = np.sum(Pxx[alpha_idx])
                self.band_powers[i, ch, 3] = np.sum(Pxx[beta_idx])

        # 归一化处理
        if self.Normalization:
            for ch in range(self.num_channels):
                for band in range(4):
                    max_val = np.max(self.band_powers[:, ch, band])
                    if max_val > 0:
                        self.band_powers[:, ch, band] /= max_val

        return True

    def plot_channel(self, channel: int):
        """绘制指定通道的频带能量变化图并返回Base64编码的图像"""
        if self.band_powers is None or channel >= self.num_channels:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        time_axis = np.arange(
            self.band_powers.shape[0]) * (self.step_size / self.Fs)

        for band in range(4):
            ax.plot(
                time_axis, self.band_powers[:, channel, band], label=self.band_names[band])

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Power' if self.Normalization else 'Power')
        ax.set_title(f'Channel {channel + 1} Band Power Dynamics')
        ax.legend()
        ax.grid(True)

        # 将图表转换为Base64编码的图像
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


class EEGAnalysisApp:
    def __init__(self):
        self.analyzer = EEGAnalyzer()
        self.create_ui()

    def create_ui(self):
        """创建用户界面"""
        with ui.header():
            ui.label('EEG数据分析工具').classes('text-2xl font-bold')

        with ui.row().classes('w-full'):
            with ui.column().classes('w-1/4 p-4 border-r'):
                ui.label('参数设置').classes('text-xl font-bold')

                with ui.card():
                    self.window_size = ui.number(
                        label='窗口大小', value=500, min=100, max=5000, step=100
                    ).bind_value(self.analyzer, 'window_size')

                    self.step_size = ui.number(
                        label='滑动步长', value=250, min=50, max=2500, step=50
                    ).bind_value(self.analyzer, 'step_size')

                    self.Fs = ui.number(
                        label='采样率(Hz)', value=500, min=100, max=2000, step=50
                    ).bind_value(self.analyzer, 'Fs')

                    self.normalization = ui.checkbox(
                        '归一化', value=True
                    ).bind_value(self.analyzer, 'Normalization')

                ui.separator()

                ui.label('文件操作').classes('text-xl font-bold')
                with ui.card():
                    self.file_upload = ui.upload(
                        label='选择EEG数据文件',
                        on_upload=self.handle_upload,
                        multiple=False,
                        # accept='.txt,.csv',
                    ).classes('w-full')

                    self.selected_file = ui.label('未选择文件').classes('w-full')

                ui.button('分析数据', on_click=self.analyze_data).classes(
                    'w-full mt-4')

            with ui.column().classes('w-2/4 p-4'):
                self.results_container = ui.column().classes('w-full')
                self.status_label = ui.label('准备就绪').classes(
                    'w-full text-sm text-gray-500')

    def handle_upload(self, e):
        """处理文件上传"""
        try:
            print(e)
            file_path = e.content
            # TODO: Copy file_path.name to the file_path
            self.selected_file.set_text(f"已选择文件: {file_path.name}")
            self.analyzer.load_data(file_path)
            self.status_label.set_text(
                f"已加载文件: {file_path.name}, 通道数: {self.analyzer.num_channels}")
        except Exception as ex:
            ui.notify(f"文件处理错误: {str(ex)}", type='negative')
            self.status_label.set_text(f"错误: {str(ex)}")

    def analyze_data(self):
        """执行数据分析"""
        if self.analyzer.data is None:
            ui.notify("请先选择数据文件", type='warning')
            return

        try:
            self.status_label.set_text("正在分析数据...")
            ui.notify("开始分析数据", type='info')

            # 计算频带能量
            success = self.analyzer.compute_band_powers()
            if not success:
                ui.notify("数据分析失败", type='negative')
                return

            # 清空结果容器
            self.results_container.clear()

            # 添加文件信息
            with self.results_container:
                ui.label(
                    f"分析结果 - {self.analyzer.current_file}").classes('text-xl font-bold')
                ui.label(
                    f"通道数: {self.analyzer.num_channels}, 窗口数: {self.analyzer.band_powers.shape[0]}")
                ui.separator()

                # 为每个通道创建图表
                for ch in range(self.analyzer.num_channels):
                    ui.label(
                        f"通道 {ch + 1} 频带能量变化").classes('text-lg font-bold mt-4')

                    # 获取图表图像
                    img_data = self.analyzer.plot_channel(ch)
                    if img_data:
                        ui.image(img_data).classes('w-full')

                    ui.separator()

            self.status_label.set_text(f"分析完成 - {self.analyzer.current_file}")
            ui.notify("分析完成", type='positive')

        except Exception as ex:
            import traceback
            traceback.print_exc()
            ui.notify(f"分析过程中出错: {str(ex)}", type='negative')
            self.status_label.set_text(f"错误: {str(ex)}")


# 启动应用
app = EEGAnalysisApp()
ui.run(title='EEG数据分析工具', port=8080)
