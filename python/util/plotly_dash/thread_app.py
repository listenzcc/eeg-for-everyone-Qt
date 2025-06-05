"""
File: thread_app.py
Author: Chuncheng Zhang
Date: 2024-06-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-26 ------------------------
# Requirements and constants
import time
from datetime import datetime

from dash import Dash, html
from threading import Thread

from omegaconf import OmegaConf

CONF = OmegaConf.load('config.yaml')


# %% ---- 2024-06-26 ------------------------
# Function and class
def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class DashApp(object):
    host = CONF.Dash.host  # 'localhost'
    port = CONF.Dash.port  # 8693

    dynamic_html = ''

    def __init__(self):
        self.app = Dash('EEG for everyone')
        self.layout()
        self.run_forever()

    def layout(self):
        self.h1 = html.H1(
            children='EEG for everyone',
            style={'textAlign': 'center'})

        self.p = html.P(
            children=f'Now: {datetime.now()}',
            style={'textAlign': 'center'})

        self.div = html.Div(
            children=[],
            # Keep the div to the center of the page
            style={'margin': 'auto', 'width': 'fit-content'}
        )

        self.app.layout = [
            self.h1,
            self.p,
            self.div
        ]

    def run_forever(self):
        Thread(target=self.serve, daemon=True).start()
        Thread(target=self.tictoc, daemon=True).start()

    def serve(self):
        self.app.run(debug=False, host=self.host, port=self.port)

    def tictoc(self, interval: float = 0.5):
        while True:
            self.p.children = f'Now: {datetime.now()}'
            time.sleep(interval)


# %% ---- 2024-06-26 ------------------------
# Play ground


# %% ---- 2024-06-26 ------------------------
# Pending


# %% ---- 2024-06-26 ------------------------
# Pending
