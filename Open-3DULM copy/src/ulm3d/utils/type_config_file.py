"""
This file contains function to check variables type of the config file.
"""


def check_type_config_file(config: dict):
    """
    Check the types of variables from the yaml config file.

    Args:
        config (dict): The yaml dictionnary to check.
    """

    if "input_var_name" in config:
        assert isinstance(config["input_var_name"], str)
    if "max_workers" in config:
        assert isinstance(config["max_workers"], int)
    if "export_extension_tracks_localizations" in config:
        assert isinstance(config["export_extension_tracks_localizations"], list)
    if "export_extension_volume" in config:
        assert isinstance(config["export_extension_volume"], list)
    if "export_volume" in config:
        assert isinstance(config["export_volume"], list)

    assert isinstance(config["volumerate"], int)

    assert isinstance(config["z_dim"], int)
    assert (
        config["z_dim"] >= 0 and config["z_dim"] < 3
    ), "Please make sur that z_dim is 0, 1, or 2."
    assert isinstance(config["voxel_size"], list)
    assert isinstance(config["origin"], list)

    assert isinstance(config["res"], int)
    assert isinstance(config["max_velocity"], int)
    assert isinstance(config["filt_mode"], str)
    if config["filt_mode"] == "SVD_bandpass":
        assert isinstance(config["bandpass_filter"], list)
        assert isinstance(config["filter_order"], int)
    if config["filt_mode"] == "SVD" or config["filt_mode"] == "SVD_bandpass":
        assert isinstance(config["svd_values"], list)
    assert isinstance(config["number_of_particles"], int)
    assert isinstance(config["min_length"], int)
    assert isinstance(config["nb_local_max"], int)
    assert isinstance(config["min_snr"], int)
    assert isinstance(config["patch_size"], list)
    assert isinstance(config["max_gap_closing"], int)
