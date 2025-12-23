import numpy as np
from loguru import logger

from ulm3d.ulm import ULM
from ulm3d.utils.load_data import load_iq


def compute_power_doppler(
    files: list,
    ulm: ULM,
    input_var_name="",
) -> np.ndarray:
    """Computes the Power Doppler from IQ data.

    Args:
        files (list): List of IQ files used to generate the Power Doppler.
        ulm (ulm3d.ulm.ULM): ULM object used to apply ULM processing methods.
        input_var_name (str): Name of the variable in the input file containing the IQ data.

    Returns:
        np.ndarray: A 3D array representing the computed Power Doppler.
    """
    logger.info(f"Compute Power Doppler on {len(files)} blocs.")
    power_doppler = np.zeros(ulm.size[:3], dtype=np.float32)
    for file in files:
        iq = load_iq(file, input_var_name)

        # Filtering.
        if ulm.filt_mode != "no_filter":
            iq = ulm.filtering(iq)  # Filtering is applied.
        power_doppler += np.sum(np.abs(iq), axis=-1)
    power_doppler = np.sqrt(power_doppler) / len(files)
    return power_doppler
