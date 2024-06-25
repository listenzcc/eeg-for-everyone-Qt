"""
File: input_dialog.py
Author: Chuncheng Zhang
Date: 2024-06-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Make Qt Dialog for temporally input

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-06-25 ------------------------
# Requirements and constants
import json
from PySide6 import QtWidgets

from . import logger


# %% ---- 2024-06-25 ------------------------
# Function and class
def require_options_with_QDialog(default_options: dict = {}, comment: str = '# Comment'):
    # Set text as default_options
    # It equals to
    # >> input_buffer = {'text': json.dumps(default_potions)}
    text = '{\n  ' + ',\n  '.join([
        f'"{k}": "{v}"' if isinstance(v, str) else f'"{k}": {v}'
        for k, v in default_options.items()]) + '\n}'
    text = f'{comment}\n{text}'
    text = text.replace('\'', '\"')
    input_buffer = {'text': text}

    # Make dialog
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle('Require frequencies')

    layout = QtWidgets.QVBoxLayout()
    text_area = QtWidgets.QTextEdit()
    layout.addWidget(text_area)
    dialog.setLayout(layout)

    text_area.setText(input_buffer['text'])

    def on_text_changed():
        # Handle input changes
        input_buffer.update(dict(
            text=text_area.document().toPlainText()
        ))

    text_area.textChanged.connect(on_text_changed)

    # Display the dialog
    dialog.exec()

    # Return input options or {} if error occurred
    try:
        text = '\n'.join([e for e in input_buffer['text'].split(
            '\n') if not e.strip().startswith('#')])
        text = text.replace(' ', '').replace('\n', '').replace('\t', '')
        logger.debug(f'Got input: {text}')
        return json.loads(text)
    except Exception as error:
        logger.error(f'{error}')
        import traceback
        traceback.print_exc()
        return {}

# %% ---- 2024-06-25 ------------------------
# Play ground


# %% ---- 2024-06-25 ------------------------
# Pending


# %% ---- 2024-06-25 ------------------------
# Pending
