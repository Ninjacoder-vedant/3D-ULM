import pathlib

import h5py
import mat73
import numpy as np
from scipy.io import loadmat


def load_iq(iq_file: str, input_var_name="") -> np.ndarray:
    ext = pathlib.Path(iq_file).suffix

    if ext == ".mat":
        # Load IQ.
        try:
            iq = loadmat(iq_file, squeeze_me=True)[input_var_name]
        except NotImplementedError:
            iq = mat73.loadmat(iq_file)[input_var_name]  # If IQ is saved in 7.3.

    elif ext == ".npy":
        iq = np.load(iq_file)

    elif ext == ".npz":
        iq = np.load(iq_file)[input_var_name]

    else:
        raise NotImplementedError(
            f"Loading iq with extension {ext} is not implemented."
        )

    return iq


def load_volume(volume_file: str) -> dict:
    ext = pathlib.Path(volume_file).suffix

    if ext == ".npz":
        data_volume = np.load(volume_file)

    elif ext == ".hdf5":
        data_volume = {}
        with h5py.File(volume_file, "r") as f:
            print("Keys: %s" % f.keys())
            for key in f.keys():
                volume = f[key][()]  # returns as a numpy array
                data_volume[key] = volume
    else:
        raise NotImplementedError(
            f"Loading volume with extension {ext} is not implemented."
        )

    return data_volume
