"""
This file contains functions that replicate MATLAB behavior.
"""

from math import ceil

import numpy as np
import scipy.signal
from loguru import logger


def smooth(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth the data as it is done in MATLAB (for basic usage of smooth).
    Code found on Stack Overflow: https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python.

    Args:
        data (np.ndarray): NumPy 2-D array containing the data to be smoothed (along the first dimension).
        window (int, optional): Smoothing window size, which must be an odd number.

    Returns:
        np.ndarray: The smoothed data.
    """
    if data.ndim < 2:
        data = np.expand_dims(data, 1)

    if data.shape[0] < window:
        logger.trace(f"array to short for conv window {window}")
        return data
    if window < 1:
        window = ceil(data.shape[0] * window)
        window = int(window - 1 + (window % 2))
    window = int(window)
    mask = np.ones((int(window), 1), dtype=int) / window

    out = scipy.signal.convolve(data, mask, "valid")

    r = np.arange(1, window - 1, 2)
    r = np.expand_dims(r, 1)
    head = np.cumsum(data[: window - 1, :], axis=0)[::2, :] / r
    tail = np.cumsum(data[:-window:-1, :], axis=0)[::2, :] / r
    tail = tail[::-1, :]
    return np.concatenate((head, out, tail), axis=0)
