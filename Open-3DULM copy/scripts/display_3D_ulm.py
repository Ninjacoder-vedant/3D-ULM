"""
Visualizer for Open-3DULM volumes (density, velocity, directivity, etc.)
Loads .npz/.hdf5 volumes generated after reconstruction and displays
a 3-axis Maximum-Intensity-Projection (MIP) for each.
"""

import argparse
import os
import sys
from tkinter import Tk, filedialog

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Open-3DULM helper to load volume files
from ulm3d.utils.load_data import load_volume


# Make axis labels smaller for compact figures
mpl.rcParams["xtick.labelsize"] = "x-small"
mpl.rcParams["ytick.labelsize"] = "x-small"


# ------------------------------------------------------------------------------
# Command-line arguments (optional)
# ------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Display 3D-ULM volumes")

    parser.add_argument(
        "-i","--input",
        type=str,
        default=None,
        help="Folder containing ULM volume files",
    )

    parser.add_argument(
        "-v","--verbose-level",
        type=int,
        default=1,
        choices=range(4),
        help="Logging level",
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="mm",
        choices=("pixel","mm"),
        help="Show axis in physical millimeters (recommended)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive figures instead of saving only",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Read volume content and metadata
# ------------------------------------------------------------------------------
def get_data(file: str):
    """
    Loads ULM volume + metadata using Open-3DULM loader.
    Returns: volume array, label string, pitch, origin
    """

    # Reads .npz/.hdf5 dictionary
    content = load_volume(file)

    mat = None
    label = ""

    # Default origin = (0,0,0) if missing
    origin = np.zeros(3)

    # Find the main ND volume array
    for key,val in content.items():
        if isinstance(val, np.ndarray):
            if val.size > 5:       # ignore very tiny arrays
                mat = val
                label = key        # name of dataset: density/velocity/etc
                break

    # "pitch" = voxel size in mm used for physical axis scaling
    if "pitch" in content:
        pitch = content["pitch"].squeeze()
    else:
        pitch = np.ones(mat.ndim)  # fallback
        logger.warning(f"Missing pitch in {file}")

    # origin = physical origin of voxel (mm)
    if "origin" in content:
        origin = content["origin"].squeeze()

    if mat is None:
        logger.warning(f"No volume found in {file}")
    else:
        logger.info(f"Volume {label} ({mat.shape}, pitch {pitch}) found in {file}")

    return mat,label,pitch,origin



# ------------------------------------------------------------------------------
# Display each volume file
# ------------------------------------------------------------------------------
def export_rendering(vol_files: list, show: bool, scale: bool):

    for i,file in enumerate(vol_files):
        logger.info(f"Render ({i+1}/{len(vol_files)})  {file}")

        mat,label,pitch,origin = get_data(file)
        if mat is None:
            continue

        # If 4D, collapse time-axis
        if mat.ndim == 4:
            mat = np.sum(mat,axis=-1)

        # Only show 3D matrices
        if mat.ndim != 3:
            continue

        # Default settings
        func             = np.mean      # projection type
        compress_power   = 1            # dynamic compression
        cmap             = "hot"

        # Pre-compute color range
        mat_proj = np.power(func(mat,axis=1),compress_power)
        clim     = np.array([np.nanmin(mat_proj),np.nanmax(mat_proj)])

        # Different behavior depending on the volume type
        if "density" in file:
            compress_power = 0.8
            clim = np.power(clim,compress_power)

        elif "doppler" in file:
            compress_power = 0.75         # increase contrast
            clim = np.power(clim,compress_power)

        elif "directivity" in file:
            compress_power = 0.8
            cmap = "twilight"             # symmetric colormap for +/- direction
            clim = np.array([-1,1])*np.max(np.abs(clim))*0.5

        elif "velocity" in file:
            cmap = "jet"                  # typical medical convention
            func = np.max                 # take max project
            clim = [0,func(mat)]

        # Prepare figure
        fig,axs = plt.subplots(1,3,figsize=(12,4))
        fig.suptitle(
            f"{label} (pitch={pitch}, compression={compress_power})",
            size="medium"
        )

        # Show projections along Z,Y,X
        for ind,ax_name in enumerate(list("ZYX")):

            # Project along dimension ind
            mat_proj = func(mat,axis=ind)

            # Project pitch and origin to a 2D plane
            pitch_proj  = np.delete(pitch,ind)
            ax_proj     = np.delete(list("ZYX"),ind)
            origin_proj = np.delete(origin,ind)

            # Extent controls axis labels
            extent = -0.5 + np.array([0,mat_proj.shape[1],mat_proj.shape[0],0])

            if scale:
                # Convert voxel → mm
                extent[:2] = extent[:2]*pitch_proj[1] + origin_proj[1]
                extent[2:] = extent[2:]*pitch_proj[0] + origin_proj[0]

            im = axs[ind].imshow(
                np.sign(mat_proj)*np.power(np.abs(mat_proj),compress_power),
                aspect="equal",
                cmap=cmap,
                clim=clim,
                extent=extent,
            )
            axs[ind].set_title(f"MIP Axis {ax_name}")

            if scale:
                axs[ind].set(xlabel=f"{ax_proj[1]} [mm]",
                             ylabel=f"{ax_proj[0]} [mm]")

        # Add colorbar to the bottom
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        cbar_ax = fig.add_axes([0.25,0.10,0.5,0.02])

        # Normalizer for colorbar
        norm = mcolors.Normalize(vmin=np.nanmin(mat),vmax=np.nanmax(mat))

        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm,cmap=cmap),
            cax=cbar_ax,
            orientation="horizontal",
        )

        # Save PNG next to data
        fig_path = os.path.join(os.path.dirname(vol_files[0]), label+".png")
        logger.info(f"Saved figure → {fig_path}")
        fig.savefig(fig_path)

        if show:
            plt.show()



# ------------------------------------------------------------------------------
# Program entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=["WARNING","INFO","DEBUG","TRACE"][args.verbose_level],
    )

    # If input folder not provided: open file browser
    if args.input is None:
        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost",1)

        volume_files = filedialog.askopenfilenames(
            initialdir=".",
            title="Select ULM volume files",
            filetypes=(("hdf5 files","*.hdf5"),
                       ("npz files","*.npz"),
                       ("all","*.*")),
            parent=root,
        )

    else:
        # Otherwise load all files in directory
        volume_files = [
            os.path.join(os.path.abspath(args.input),d)
            for d in os.listdir(args.input)
            if not d.endswith(".png") and not d.endswith(".npy")
        ]

    logger.success(
        f"{len(volume_files)} volume files found in {os.path.dirname(volume_files[0])}"
    )

    # Process
    export_rendering(
        vol_files=volume_files,
        show=args.show,
        scale=args.scale,
    )
