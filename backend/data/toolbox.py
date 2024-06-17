"""
File: toolbox.py
Author: Chuncheng Zhang
Date: 2024-02-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The tools for data operations.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-21 ------------------------
# Requirements and constants
import hashlib


# %% ---- 2024-02-21 ------------------------
# Function and class
def md5_encode(string: str) -> str:
    """
    Returns the MD5 hash of the given string.

    Args:
        string (str): The string to be encoded.

    Returns:
        str: The MD5 hash of the string.
    """
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()


# %% ---- 2024-02-21 ------------------------
# Play ground


# %% ---- 2024-02-21 ------------------------
# Pending
if __name__ == "__main__":
    string = "Hello, World!"
    md5 = md5_encode(string)
    print(f"Md5 encode: {string} -> {md5}")


# %% ---- 2024-02-21 ------------------------
# Pending
