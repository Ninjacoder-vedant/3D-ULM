"""
=======================
Open_3DULM main script
=======================

This script runs the entire 3D-ULM pipeline:

✔ load IQ data
✔ filtering (SVD/Bandpass)
✔ super-localization
✔ tracking
✔ export localizations + tracks
✔ compute global 3D density/velocity volumes
✔ save 3D volumetric outputs

Input:
    - IQ files (beamformed ultrasound data)
    - YAML config describing all parameters

Output:
    - per-block raw + interpolated tracks
    - global dense 3-D volumes
    - power Doppler
"""

import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tkinter import Tk, filedialog

import yaml
from loguru import logger
from tqdm import tqdm

import ulm3d.ulm
import ulm3d.utils
import ulm3d.utils.export
import ulm3d.utils.power_doppler
import ulm3d.utils.render
import ulm3d.utils.type_config_file
from ulm3d.utils.create_archi_export import (create_archi_export,
                                             increment_config_folder)
from ulm3d.utils.load_data import load_iq


# -----------------------------------------------------------
# Parse optional CLI arguments
# -----------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="3D ULM reconstruction")

    parser.add_argument("--config-file",type=str,default=None,
                        help="Config YAML")

    parser.add_argument("-i","--input",type=str,default=None,
                        help="Directory of IQ .mat files")

    parser.add_argument("-o","--output",type=str,default=None,
                        help="Where to store results")

    parser.add_argument("-v","--verbose-level",type=int,default=1,
                        choices=range(4),help="Verbosity")

    parser.add_argument("--workers",type=int,default=None,
                        help="Parallel workers (override config)")

    return parser.parse_args()



# -----------------------------------------------------------
#  Process ONE IQ block (filter → localization → tracking)
# -----------------------------------------------------------
def compute_bloc(ulm_pipeline,iq_files,input_var_name,export_parameters,index):
    """
    Runs the ULM pipeline on one file.

    Steps:
      - load IQ .mat
      - filter
      - localize microbubbles
      - track
      - export tracks
    """

    # Load complex IQ data
    iq = load_iq(iq_files[index], input_var_name)
    

    # Apply chosen filtering mode (SVD/Bandpass/etc)
    if ulm_pipeline.filt_mode != "no_filter":
        iq_before_loc = ulm_pipeline.filtering(iq)
    else:
        iq_before_loc = iq

    # Perform 3D super-localization
    localizations = ulm_pipeline.super_localization(iq_before_loc)

    # Save localization file if requested
    if "localizations" in export_parameters and localizations.shape[0]>0:
        ulm3d.utils.export.export_locs(
            index,localizations,export_parameters["localizations"]
        )
    elif localizations.shape[0]==0:
        logger.warning(f"No localizations detected in bloc {index}.")

    # Track microbubbles across frames
    tracks = ulm_pipeline.create_tracks(localizations)

    # Save raw/interpolated tracks if requested
    if "tracks" in export_parameters and tracks[1].shape[0]>0:
        ulm3d.utils.export.export_tracks(index,tracks,export_parameters["tracks"])
    elif tracks[1].shape[0]==0:
        logger.warning(f"No tracks detected in bloc {index}.")



# -----------------------------------------------------------
#  MAIN PIPELINE (multiple blocks)
# -----------------------------------------------------------
def run(config_file,iq_files,output_dir,workers):
    """
    Executes 3D-ULM on all blocks and aggregates results
    """

    # Load YAML settings
    with open(config_file) as stream:
        config = yaml.safe_load(stream)

    logger.debug(f"Loaded config:\n{yaml.dump(config)}")

    # If user did not specify workers → use config
    if workers is None:
        workers = config["max_workers"]
        logger.info(f"Using default workers = {workers}")

    # Type checking of config entries
    ulm3d.utils.type_config_file.check_type_config_file(config)

    # Store IQ folder in config
    config["IQ_folder_path"] = os.path.dirname(iq_files[0])

    # Create folder architecture
    export_parameters = create_archi_export(output_dir, config)

    # Instantiates ULM pipeline class
    ulm = ulm3d.ulm.ULM(iq_files=iq_files, **config)

    # Name of variable inside .mat
    input_var_name = config.get("input_var_name","")

    # -------------------------------------------------------
    # Optional: compute power Doppler
    # -------------------------------------------------------
    if "power_doppler" in config["export_volume"]:
        power_doppler = ulm3d.utils.power_doppler.compute_power_doppler(
            iq_files[:min(len(iq_files),2)], # only a few frames needed
            ulm,
            input_var_name,
        )
        # Save Doppler volume
        ulm3d.utils.render.save_output(
            os.path.join(output_dir,"volume","power_doppler"),
            {
                "power_doppler":power_doppler,
                "pitch":ulm.scale[:3],
                "origin":ulm.origin[:3],
            },
            export_parameters["3D_rendering"]["export_extension_volume"],
        )


    # -------------------------------------------------------
    # Process ALL IQ blocks
    # -------------------------------------------------------
    if workers==1:
        logger.info("Processing sequentially")

        for ind,_ in enumerate(tqdm(iq_files)):
            compute_bloc(ulm,iq_files,input_var_name,export_parameters,ind)

    else:
        # Use multiprocessing
        workers = min(workers,cpu_count())
        logger.info(f"Parallel with {workers} workers")

        with ProcessPoolExecutor(workers) as executor:
            with tqdm(total=len(iq_files)) as pbar:
                for _ in executor.map(
                    functools.partial(
                        compute_bloc,ulm,iq_files,input_var_name,export_parameters
                    ),
                    range(len(iq_files)),
                ):
                    pbar.update()

    # -------------------------------------------------------
    # After all tracks exported → build global 3D volumes
    # -------------------------------------------------------
    if "3D_rendering" in export_parameters:
        ulm3d.utils.render.rendering_3d(ulm,export_parameters)

    logger.success(f"Finished. Results saved to {output_dir}")



# -----------------------------------------------------------
# Script entry point
# -----------------------------------------------------------
if __name__=="__main__":
    args = parse_arguments()
    print(args)

    # Setup logging style
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg,end=""),
        colorize=True,
        level=["WARNING","INFO","DEBUG","TRACE"][args.verbose_level],
    )

    # -------------------------------------------------------
    # Load config YAML
    # -------------------------------------------------------
    if args.config_file is None:
        root = Tk()
        root.withdraw()
        config_file_path = filedialog.askopenfilename(
            initialdir="config/",
            title="Select config file",
            filetypes=(("yaml","*.yaml"),("all","*.*")),
        )
    else:
        config_file_path = os.path.abspath(args.config_file)

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError("Config YAML missing!")

    logger.info(f"Config file: {config_file_path}")

    # -------------------------------------------------------
    # Load IQ file list (.mat)
    # -------------------------------------------------------
    iq_files = []
    if args.input is None:
        root = Tk()
        root.withdraw()
        iq_files = filedialog.askopenfilenames(
            initialdir=".",
            title="Select IQ files",
            filetypes=(("mat","*.mat"),("npz","*.npz"),("npy","*.npy"),("all","*.*")),
        )
    else:
        for file in os.listdir(os.path.abspath(args.input)):
            if file.endswith(".mat"):
                iq_files.append(os.path.join(os.path.abspath(args.input),file))

    if len(iq_files)==0:
        raise FileNotFoundError("No IQ files found!")

    logger.info(f"{len(iq_files)} IQ files selected")

    # -------------------------------------------------------
    # Output folder creation
    # -------------------------------------------------------
    if args.output is None:
        output_dir = filedialog.askdirectory(initialdir=".")
    else:
        output_dir = os.path.abspath(args.output)

    if not output_dir:
        raise FileNotFoundError("Please choose output folder.")

    # Make folder unique (_config_0, _config_1, ...)
    output_dir = increment_config_folder(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    logger.info(f"Output directory = {output_dir}")

    # -------------------------------------------------------
    # RUN ULM PIPELINE
    # -------------------------------------------------------
    run(
        config_file=config_file_path,
        iq_files=iq_files,
        output_dir=output_dir,
        workers=args.workers,
    )
