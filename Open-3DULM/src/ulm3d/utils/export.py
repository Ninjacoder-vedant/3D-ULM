"""
This file is used to save localizations and 3D tracks in csv format.
"""

import os

import h5py
import numpy as np


def export_csv(
    data: np.ndarray,
    output_path: str,
):
    """
    This function is used to save localizations and 3D tracks in csv format.

    Args:
        data (np.ndarray): The array to save.
        output_path (str): The path where the file will be saved.
    """

    # Flatten the structured array to 2D.
    flattened_data = np.hstack(
        [data[name].reshape(data.shape[0], -1) for name in data.dtype.names]
    )

    # Create the header (add a field for each element in position or speed).
    header = []
    header_coords = ["0", "1", "2"]
    for name in data.dtype.names:
        if data.dtype[name].shape:
            header.extend(
                [f"{name}_{header_coords[i]}" for i in range(data.dtype[name].shape[0])]
            )
        else:
            header.append(name)

    # Save to CSV.
    np.savetxt(
        output_path,
        flattened_data,
        delimiter=";",
        header=";".join(header),
        comments="",
        fmt="%s",
    )


def export_locs(
    index: int,
    localizations: np.ndarray,
    export_parameters: dict,
):
    """
    This function exports the localizations. According to the config file, the localizations can be saved in:
        - HDF5 format (.hdf5)
        - Numpy format (.npy)
        - CSV format (.csv)

    Args:
        index (int): The index of the current IQ being processed.
        localizations (np.ndarray): The localizations to export.
        export_parameters (dict): Parameters for exporting the localizations. Includes:
            - "folder_output": The path to save the localizations.
            - "export_extension": Determines the format to save the localizations (supported formats: "npy", "csv", "mat").
    """
    if "npz" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "npz")
        np.savez(tracks_output_path + f"\\locs_{index:04}", localizations=localizations)

    if "hdf5" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "hdf5")
        with h5py.File(tracks_output_path + f"\\locs_{index:04}" + ".hdf5", "w") as f:
            f.create_dataset("localizations", data=localizations)

    if "csv" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "csv")
        export_csv(localizations, tracks_output_path + f"\\locs_{index:04}" + ".csv")


def export_tracks(
    index: int,
    tracks: np.ndarray,
    export_parameters: dict,
):
    """
    This function exports the tracks. According to the config file, the tracks can be saved in:
        - HDF5 format (.hdf5)
        - Numpy format (.npy)
        - CSV format (.csv)

    Args:
        index (int): The index of the current IQ being processed.
        tracks (np.ndarray): The tracks to export.
        export_parameters (dict): Parameters for exporting the tracks. Includes:
            - "folder_output": The path to save the tracks.
            - "export_extension": Determines the format to save the tracks (supported formats: "npy", "csv", "mat").
    """
    if "npz" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "npz")
        np.savez(
            tracks_output_path + f"\\tracks_{index:04}",
            interp_tracks=tracks[0],
            raw_tracks=tracks[1],
        )

    if "hdf5" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "hdf5")
        with h5py.File(tracks_output_path + f"\\tracks_{index:04}" + ".hdf5", "w") as f:
            f.create_dataset("interp_tracks", data=tracks[0])
            f.create_dataset("raw_tracks", data=tracks[1])

    if "csv" in export_parameters["export_extension"]:
        tracks_output_path = os.path.join(export_parameters["folder_output"], "csv")
        export_csv(
            tracks[0], tracks_output_path + f"\\tracks_{index:04}_interp" + ".csv"
        )
        export_csv(tracks[1], tracks_output_path + f"\\tracks_{index:04}_raw" + ".csv")
