"""
This file is used to create the architecture for exporting data and parameters of the export.
"""

import os
from datetime import datetime

import yaml
from loguru import logger


def increment_config_folder(dir: str):
    """Check config_id and increment id"""
    folder_names = next(os.walk(dir))[1]
    ind = 0
    for i in range(len(folder_names)):
        try:
            try_value = folder_names[i].split("config_")[1]
            if int(try_value) > ind:
                ind = int(try_value)
        except:
            continue
    dir = os.path.join(dir, f"config_{ind + 1}")
    return dir


def create_archi_export(output_dir, config: dict) -> dict:
    """
    This function creates folders for localizations, tracks and 3D rendering volume, based on the config file provided by the user.
    It returns a dictionary that contains all parameters needed to export data from the ULM pipeline.

    Args:
        output_dir (str): Output folder
        config (dict): The data from the YAML config file.

    Returns:
        dict: A dictionary that can contain the following fields:
            - localizations: if localizations have to be saved.
            - tracks: if tracks have to be saved.
            - 3D_rendering: if the user wants to export volumes for visualization.
        For each field, there is a folder_output which is the location to save localizations, tracks, or volumes for 3D rendering.
        "export_extension" is used for localizations and tracks to determine in which format they have to be saved (supported formats: "npy", "csv", "mat").
        "export_volume" is used to determine the mode of 3D rendering volume (supported mode: "density", "velocity", "directivity").

    """
    os.makedirs(output_dir, exist_ok=True)

    # Init dict which will be returned.
    export_params = {"output_dir": output_dir}

    # Iteration for each type of export (localizations and tracks).
    for export_type in ["localizations", "tracks"]:
        if "export_extension_tracks_localizations" in config:
            output_dir_type = os.path.join(output_dir, export_type)
            export_params[export_type] = {
                "folder_output": output_dir_type,
                "export_extension": config["export_extension_tracks_localizations"],
            }
            os.makedirs(output_dir_type, exist_ok=True)

        logger.trace(f"Create dir {output_dir_type}")
        # Iteration for each type of data for each type of export (csv, mat, npy).
        for extension in config["export_extension_tracks_localizations"]:
            os.makedirs(os.path.join(output_dir_type, extension), exist_ok=True)

    # Create folder for volume export if it is required by the yaml config file.
    if "export_volume" in config:
        output_dir_type = os.path.join(output_dir, "volume")
        export_params["3D_rendering"] = {
            "folder_output": output_dir_type,
            "export_volume": config["export_volume"],
            "export_extension_volume": config["export_extension_volume"],
        }
        logger.trace(f"Create dir {output_dir_type}")
        os.makedirs(output_dir_type, exist_ok=True)

    # Export yaml config file to keep what have been used to generate ULM 3D for this particular export.

    config["output_folder"] = output_dir
    config["datestr"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(os.path.join(output_dir, f"config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.success(f"Output folders created at {output_dir}")
    return export_params
