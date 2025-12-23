"""
This file contains functions to export volumes.
"""

import os
from glob import glob

import h5py
import numpy as np
import tqdm
from loguru import logger

import ulm3d.ulm


def save_output(file: str, output_dict: dict, extension_parameters: list):
    logger.debug(f"Saving {output_dict.keys()} in {file}")
    if "npy" in extension_parameters:
        key = list(output_dict.keys())[0]
        np.save(file + ".npy", output_dict[key])

    if "npz" in extension_parameters:
        np.savez(file + ".npz", **output_dict)

    if "hdf5" in extension_parameters:
        with h5py.File(file + ".hdf5", "w") as f:
            for key in output_dict.keys():
                f.create_dataset(key, data=output_dict[key])


def convert_track_to_matrix_incides(
    pos: np.ndarray,
    vel: np.ndarray,
    size_matout: np.ndarray,
    z_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The function preprares tracks (scaled in voxel) before accumating in outputs volumes.

    Volumes can be exported in different types:
        - pos: position in the output volume.
        - vel: 3D velocimetry.
        - size_matout: shape of the output volume
        - z_dim: dimension for z distinction.

    Args:
        - pos: position in the output volume.
        - vel_norm: velocity in the output volume.
        - z_comp_vel: velocity z projection.
    """

    pos = pos.astype(np.int16)
    # Find the direction and the velocity of microbubble.
    z_comp_vel = np.sign(vel[:, z_dim])
    vel_norm = np.linalg.norm(vel, ord=2, axis=1)

    # Remove microbubbles when indexes are out of range.
    keep_ind = np.logical_and(
        np.all(pos > 0, axis=1),
        np.all(pos < size_matout, axis=1),
    )
    # Apply mask to remove velocity and direction out of range.
    z_comp_vel = z_comp_vel[keep_ind]
    vel_norm = vel_norm[keep_ind]
    pos = pos[keep_ind]

    # Apply a mask to exclude out-of-range indexes, then flatten the array to identify unique indexes, ensuring each voxel is tracked only once.
    ravel_index = np.ravel_multi_index(
        [pos[:, i] for i in range(3)],
        dims=size_matout,
        order="F",
    )
    _, ii = np.unique(ravel_index, return_index=True)
    pos = pos[ii]
    z_comp_vel = z_comp_vel[ii]
    vel_norm = vel_norm[ii]
    return pos, vel_norm, z_comp_vel


def rendering_3d(
    ulm: ulm3d.ulm.ULM,
    dict_export_parameters: dict,
):
    """
    The function to export volumes for visualization.

    Volumes can be exported in different types:
        - density: See density of the vessels in the entire volume.
        - velocity: Apply 3D vector norm for velocity of microbubbles in the vessels.
        - directivity: Add the velocity vector in the axial direction (z axis).

    Args:
        ulm (ULM): The ULM object with all parameters.
        dict_export_parameters (dict): The dictionary containing all parameters to export the volumes.
    """
    logger.info("ULM rendering export")

    # -------------------------------------------------------------------
    # 1. access parameters for rendering and list all interpolated tracks
    # -------------------------------------------------------------------

    # Export settings defined earlier (folder, extensions, etc)
    export_param = dict_export_parameters["3D_rendering"]

    # Look inside ".../tracks/npz/" for all interpolated track files
    files = glob(
        os.path.join(dict_export_parameters["tracks"]["folder_output"], "npz", "*.npz")
    )
    files.sort()                         # chronological order by index

    # If no tracks exist → nothing to render
    if len(files) == 0:
        logger.warning("No tracks detected. Rendering has been stopped.")
        return

    logger.info(f"{len(files)} track files found.")


    # -------------------------------------------------------------------
    # 2. Compute the final super-resolution grid properties
    # -------------------------------------------------------------------
    # radius to crop the localization patch around PSF (half FWHM basically)
    crop_side = np.ceil(ulm.fwhm / 2)

    # super-resolution voxel pitch
    # note: they use only z-pitch scaled by res (supersampling factor)
    scale_out = np.ones(3) * ulm.scale[2] / ulm.res

    # compute output volume size:
    # (original grid – cropping) converted from mm → super-resolution voxels
    size_out = np.fix(
        (ulm.size[:-1] - 2 * crop_side) * ulm.scale[:-1] / scale_out
    ).astype(np.int16)

    # output physical origin of the super-resolution grid
    origin_out = ulm.origin[:3] + (-0.5 + crop_side) * ulm.scale[:-1]

    logger.info(
        f"Super-resolution matrix size {size_out} pitch {scale_out[0]} "
        f"(low res {ulm.size[:-1]}, pitch {ulm.scale[-1]:.3f})"
    )


    # -------------------------------------------------------------------
    # 3. Initialize final 3D matrices
    # -------------------------------------------------------------------
    # integer count (microbubble visits)
    density = np.zeros(size_out, dtype=np.int16)

    # saturation curve (fraction of volume already filled per iteration)
    saturation_curve = np.zeros(len(files))

    # directionality volume (vz signed)
    if "directivity" in export_param["export_volume"]:
        vel_norm_z_signed = np.zeros(size_out, dtype=np.float32)

    # velocity magnitude volume
    if "velocity" in export_param["export_volume"]:
        vel_norm = np.zeros(size_out, dtype=np.float32)


    # -------------------------------------------------------------------
    # 4. Iterate over every track file and accumulate contributions
    # -------------------------------------------------------------------
    for ind, f in enumerate(tqdm.tqdm(files, desc="Filling SR matrices")):

        # load interpolated tracks from npz
        tracks = np.load(f)["interp_tracks"]

        # loop over unique trajectory IDs in that file
        for j in range(tracks["track_ind"].max() + 1):

            track = tracks[tracks["track_ind"] == j]
            pos   = track["pos"]

            # convert mm → superresolution voxel indices
            # + shift origin to super-res grid
            pos = np.round((pos - origin_out) / scale_out) + 1

            # convert to valid voxel indices + extract velocity and direction
            pos, vel, dir_z = convert_track_to_matrix_incides(
                pos, track["velocity"], size_out, ulm.z_dim
            )

            # accumulate into 3D matrices
            for k in range(pos.shape[0]):
                # count density (mb passage)
                density[pos[k, 0], pos[k, 1], pos[k, 2]] += 1

                # directivity = signed velocity along z
                if "directivity" in export_param["export_volume"]:
                    vel_norm_z_signed[pos[k, 0], pos[k, 1], pos[k, 2]] += (
                        vel[k] * dir_z[k]
                    )

                # velocity magnitude
                if "velocity" in export_param["export_volume"]:
                    vel_norm[pos[k, 0], pos[k, 1], pos[k, 2]] = vel[k]

        # fraction of non-zero voxels (saturation) for this iteration
        saturation_curve[ind] = np.sum(density > 0)


    # Normalize saturation 0–1
    saturation_curve = saturation_curve / np.prod(density.shape)

    logger.info("Saving outputs matrix")


    # -------------------------------------------------------------------
    # 5. average velocity only where density > 0
    # -------------------------------------------------------------------
    mask = density > 0


    # -------------------------------------------------------------------
    # 6. Save requested output volumes (density, directivity, velocity)
    # -------------------------------------------------------------------

    if "density" in export_param["export_volume"]:
        print("Yoo")
        save_output(
            os.path.join(export_param["folder_output"], "density"),
            {
                "density": density,
                "pitch": scale_out,
                "res": ulm.res,
                "origin": origin_out,
            },
            export_param["export_extension_volume"],
        )

    if "directivity" in export_param["export_volume"]:
        # average signed velocity by count
        vel_norm_z_signed[mask] = vel_norm_z_signed[mask] / density[mask]

        save_output(
            os.path.join(export_param["folder_output"], "directivity"),
            {
                "directivity": vel_norm_z_signed,
                "pitch": scale_out,
                "res": ulm.res,
                "origin": origin_out,
                "z_dim": ulm.z_dim,
            },
            export_param["export_extension_volume"],
        )

    if "velocity" in export_param["export_volume"]:
        # average velocity magnitude
        vel_norm[mask] = vel_norm[mask] / density[mask]

        save_output(
            os.path.join(export_param["folder_output"], "velocity"),
            {
                "velocity": vel_norm,
                "pitch": scale_out,
                "res": ulm.res,
                "origin": origin_out,
            },
            export_param["export_extension_volume"],
        )

    # saturation curve vs time
    if "saturation_curve" in export_param["export_volume"]:

        save_output(
            os.path.join(export_param["folder_output"], "saturation_curve"),
            {
                "saturation_curve": saturation_curve,
                "res": ulm.res,
                "pitch": scale_out,
            },
            export_param["export_extension_volume"],
        )


    logger.success("Rendering successfully ended.")

