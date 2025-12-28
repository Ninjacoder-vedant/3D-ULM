"""This file contains ULM class."""

from typing import Tuple

import numpy as np
from loguru import logger
from peasyTracker import SimpleTracker
from scipy.ndimage import maximum_filter
from scipy.signal import butter, convolve, lfilter

from ulm3d.loc.radial_symmetry_center import radial_symmetry_center_3d
from ulm3d.utils.load_data import load_iq
from ulm3d.utils.matlab_tool import smooth
import time


class ULM:
    """
    The ULM class which contains ULM parameters and methods to apply the pipeline.

    Attributes:
        res (int): The hypotetical super-resolution factor.
        origin (int): Reconstruction grid origin in each directions.
        voxel_size (int): Voxel size in each directions [µm].
        z_dim (int): Dimension in the IQ where z is located [1,2,3].
        size (np.ndarray): Input size of raw beamformed data (IQ).
        scale (np.ndarray): Scaling for each dimensions [mm mm mm s].
        svd_tresh (np.ndarray): The range of singular values preserved for SVD filter.
        filter_order (int): The order of the bandpass filter.
        filter_fs (float): Sampling frequency [Hz] (volumerate).
        filter_fc (float): Cuttof frequency for bandpass filter [Hz].
        filter_num (np.ndarray): Numerator (b) denominator of the butterworth filter.
        filter_den (np.ndarray): Denominator (a) polynomials of the butterworth filter.
        filt_mode (str): filtering mode.
            filt_mode can be:
                - 'no_filter': nofilter.
                - 'SVD': only SVD.
                - 'SVD_bandpass': SVD + bandpass.
        number_of_particles (int): The number of microbubbles to be localized in every frame.
        fwhm (np.ndarray): Size of the PSF for localization.
        min_snr (int): The minimum SNR value for a microbubble to be accepted [dB].
        patch_size (np.ndarray): The size of the 3D kernel where the local SNR is computed.
        nb_local_max (int): The maximum number of allowed microbubbles inside a patch (fwhm^3).
        max_gap_closing (int): The maximum number of frames that the microbubble is allowed to disappear and reappear (PeasyTracker parameter).
        max_linking_distance (int): Maximum linking distance between two frames to reject pairing, in super-resolved voxels. Should be between 20 to 40 SRvoxels (PeasyTracker parameter ).
        min_length (int): The minimum number of frames a microbubble must be tracked for it to be accepted.

    Methods:
        -filtering: Filter the IQ data to remove the tissue signal, tissue motion as well as the skull signal to enhance microbubbles signal.
        -super_localization: Superlocalization function to pinpoint the microbubble's position with sub-wavelength precision.
        -create_tracks: 3D Tracking function. The 3D tracking algorithm is a library Python-adapted ('PeasyTracker') version of the open-source 'SimpleTracker' [Tinevez, J.-Y. et al. 2017].
    """

    def __init__(
        self,
        res: int,
        max_velocity: int,
        svd_values: list,
        filter_order: int,
        bandpass_filter: list,
        filt_mode: str,
        number_of_particles: int,
        nb_local_max: int,
        fwhm: list,
        min_snr: int,
        patch_size: tuple,
        min_length: int,
        max_gap_closing: int,
        z_dim: int,
        volumerate: int,
        voxel_size: list,
        origin: int,
        iq_files: list,
        input_var_name="",
        **kwargs,
    ) -> None:
        """To init ULM class.

        Args:
            volumerate (int): Number of volumes acquiried per second.
            z_dim (int): Dimension in the IQ where z is located [1,2,3].
            voxel_size (float): Voxel size in each directions [mm].
            origin (list): Reconstruction grid origin in each directions [mm].
            res (int): The hypotetical super-resolution factor.
            max_velocity (int): The max speed of the particle to track [mm/s]. (To be converted to maximum linking distance).
            bandpass_filter (int): Cutoffs for the bandpass filter.
            svd_value (int): The singular values that will be kept after applying SVD filtering.
            number_of_particles (int): The number of microbubbles to be localized in every frame.
            min_length (int): The minimum number of frames a microbubble must be tracked for it to be accepted.
            filter_order (int): The order of the bandpass filter.
            filt_mode (str): Filtering mode.
                filt_mode can be:
                    - 'no_filter': nofilter.
                    - 'SVD': only SVD.
                    - 'SVD_bandpass': SVD + bandpass.
            fwhm (list): Size of the PSF for localization in each directions.
            nb_local_max (int): The maximum number of allowed microbubbles inside a patch (fwhm^3).
            min_snr (int):The minimum SNR value for a microbubble to be accepted [dB].
            patch_size (np.ndarray): The size of the 3D kernel where the local SNR is computed.
            max_gap_closing (int): The maximum number of frames that the microbubble is allowed to disappear and reappear (simpletracker parameter).
            input_var_name (str): Name of the input variable when IQ are loaded.
            iq_files (list): The list of the paths of the IQ. Used to load an IQ to determine the shape of IQ.
        """

        logger.info("Initializing ULM pipeline...")
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Size parameters.
        self.res = res
        self.origin = np.array(origin)
        self.z_dim = z_dim

        iq = load_iq(iq_files[0], input_var_name)
        self.size = iq.shape
        self.scale = np.append(np.array(voxel_size), 1 / volumerate)

        # Filtering parameters.
        self.svd_tresh = np.array(svd_values, dtype=np.uint16)
        self.filter_order = filter_order

        if filt_mode == "SVD_bandpass":
            self.filter_fs = volumerate
            self.filter_fc = np.array(
                bandpass_filter,
                dtype=np.uint16,
            )
            self.filter_num, self.filter_den = butter(
                self.filter_order, self.filter_fc / (self.filter_fs / 2), "bandpass"
            )
            logger.debug(
                f"Set bandpass filter: order {filter_order}, fc {self.filter_fc}"
            )

        self.filt_mode = filt_mode

        # Detection parameters.
        self.number_of_particles = number_of_particles
        self.fwhm = np.array(fwhm)
        self.min_snr = min_snr
        self.patch_size = np.array(patch_size)
        self.nb_local_max = nb_local_max

        # Tracking parameters.
        self.max_gap_closing = max_gap_closing
        logger.info(f"max_gap_closing: { self.max_gap_closing}")

        max_link_dist = max_velocity / volumerate / voxel_size[self.z_dim] * self.res
        self.max_linking_distance = np.round(max_link_dist)
        mld_mm = self.max_linking_distance / self.res * np.mean(self.scale[:3])
        logger.info(
            f"max_linking_distance_dist: {mld_mm} (~{mld_mm/np.mean(self.scale[:3])} voxel)"
        )
        self.min_length = min_length
        logger.info(f"min_track_len: { self.min_length}")

    def filtering(self, iq: np.ndarray) -> np.ndarray:
        """
        Filter the IQ data to remove the tissue signal, tissue motion as well as the skull signal to enhance microbubbles signal.

        Args:
            iq (np.ndarray): The IQ data to be filtered.

        Returns:
            np.ndarray: The filtered IQ data.
        """
        start_time = time.time()
        iq = iq.astype("complex128")
        # Extract shape of IQ.

        # Reshape IQ in 2D to SVD filtering (Casorati matrix space x time).
        iq_shape = iq.shape
        iq = np.reshape(iq, (-1, iq_shape[-1]), order="F")

        # Apply SVD.
        u, _, _ = np.linalg.svd(np.matmul(np.conj(iq.T), iq), full_matrices=True)
        v = np.matmul(iq, u)

        logger.debug(f"SVD filtering: keep {self.svd_tresh} over {iq_shape[3]} frames.")
        # Compute array filtered in 2D.
        iq_filtered = np.matmul(
            v[:, self.svd_tresh[0] - 1 : self.svd_tresh[1]],
            np.conj(u[:, self.svd_tresh[0] - 1 : self.svd_tresh[1]]).T,
        )

        # Get original shape.
        iq_filtered = np.reshape(iq_filtered, (iq_shape), order="F")

        # Apply bandpass filter if it is needed
        if self.filt_mode == "SVD_bandpass":
            iq_filtered = lfilter(self.filter_num, self.filter_den, iq_filtered)
        end_time = time.time()
        logger.debug(f"[filtering] Time taken: {end_time-start_time} sec")
        return iq_filtered

    def super_localization(
        self,
        iq: np.ndarray,
        type_name: str = "float",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Superlocalization function to pinpoint the microbubble's position with sub-wavelength precision.

        Args:
            iq (np.ndarray): The filtered IQ data.
            type_name (str, optional): The type of the IQ matrix. Defaults to "float".

        Returns:
            np.ndarray: A structured array containing matrix with the positions of the superlocalization in sub-voxel. The different fields are:
                - snr: The local SNR of the microbubble.
                - pos: The sub-wavelength position of the microbubble (sub-voxel).
                - frame_no: The index of the frame where the microbubble is located.
        """
        start_time = time.time()
        iq = np.abs(np.asarray(iq))
        mask, intensity = get_intensity_matrix(iq, self.fwhm, type_name)

        (
            local_contrast,
            final_mask,
            linear_ind_mb_detection,
            index_frames,
        ) = get_local_contrast(
            iq,
            mask,
            intensity,
            min_snr=self.min_snr,
            patch_size=self.patch_size,
        )

        # Verify if the number of detections exceeds the expected number of particles.
        # If there are too many detections, remove the particles with the lowest local contrast values.
        if np.size(index_frames) / self.size[-1] > self.number_of_particles:
            logger.debug(
                f"Too many detection, reducing to {self.number_of_particles} per frame."
            )
            for frame in range(self.size[-1]):
                idx_current_frame = np.unravel_index(
                    linear_ind_mb_detection[index_frames == frame],
                    local_contrast.shape,
                    order="F",
                )
                local_contrast_i = local_contrast[
                    idx_current_frame
                ]  # value for each frame.
                if np.size(local_contrast_i) > self.number_of_particles:
                    # Sort in descending order.
                    sort_idx_current_frame = np.argsort(local_contrast_i)[::-1]
                    # remove bubble i ind.
                    local_contrast_i[
                        sort_idx_current_frame[self.number_of_particles :]
                    ] = 0

                    # Keep only the linear index that match condition of number of particles.
                    linear_ind_mb_detection[index_frames == frame] = (
                        linear_ind_mb_detection[index_frames == frame]
                        * (local_contrast_i > 0)
                    )

        linear_ind_mb_detection = linear_ind_mb_detection[linear_ind_mb_detection != 0]
        logger.debug(f"{len(linear_ind_mb_detection)} maxima kept")
        ind_4D = np.unravel_index(
            linear_ind_mb_detection, local_contrast.shape, order="F"
        )
        list_snr = local_contrast[ind_4D]
        indices_4d = np.unravel_index(
            linear_ind_mb_detection,
            final_mask.shape,
            order="F",
        )
        index_mask = np.stack(indices_4d)
        index_frames = indices_4d[-1]

        index_frames = np.sort(index_frames)
        ind_t = np.sort(np.argsort(index_frames))
        index_mask = index_mask[:, ind_t]

        # Creating FWHM models.
        vectfwhm = [None] * 3
        for i, h_fwhm in enumerate(np.ceil(self.fwhm / 2).astype(int)):
            v_fwhm = np.arange(-h_fwhm, h_fwhm + 1).astype(np.int64)
            v_shape = [1] * 3
            v_shape[i] = -1
            vectfwhm[i] = np.reshape(v_fwhm, v_shape)

        # Localization.
        intensity_center = np.zeros((index_mask.shape[1]), dtype=type_name)
        pos = np.zeros_like(index_mask[:3, :], dtype=np.float32)

        for iscat in range(index_mask.shape[1] - 1, -1, -1):
            intensity_roi = np.absolute(
                iq[
                    index_mask[0, iscat] + vectfwhm[0],
                    index_mask[1, iscat] + vectfwhm[1],
                    index_mask[2, iscat] + vectfwhm[2],
                    index_mask[3, iscat],
                ]
            )
            if index_frames[iscat] < iq.shape[3]:
                # Remove frames from index_frames[iscat]+1 to the end.
                iq = iq[..., : index_frames[iscat] + 1]
            mask = maximum_filter(intensity_roi, footprint=np.ones((3, 3, 3)))
            mask = mask == intensity_roi

            # Remove localizations that exceed a specified number of local maxima criteria, keeping those with the highest SNR.
            if np.count_nonzero(mask) > self.nb_local_max:
                index_mask = np.delete(index_mask, iscat, axis=1)
                pos = np.delete(pos, iscat, axis=1)
                intensity_center = np.delete(intensity_center, iscat)
                continue
            sub_pos = radial_symmetry_center_3d(intensity_roi)
            pos[:, iscat] = index_mask[:3, iscat] + np.asarray(sub_pos)
            intensity_center[iscat] = list_snr[iscat]

        if pos.shape[1] == 0:
            logger.warning("0 microbubble located !")
            return False, None

        new_struct = []
        for i in range(pos.shape[1]):
            new_struct.append(
                (
                    intensity_center[i],
                    pos[:, i],
                    index_mask[3, i],
                )
            )

        structured_localizations = np.array(
            new_struct,
            dtype=[
                ("snr", float),
                ("pos", float, 3),
                ("frame_no", int),
            ],
        )
        end_time = time.time()
        logger.debug(f"[super_localization] Time taken: {end_time-start_time} sec")
        return structured_localizations

    def create_tracks(self, localizations: np.ndarray) -> np.ndarray:
        """
        3D Tracking function. The 3D tracking algorithm is a library Python-adapted ('PeasyTracker') version of the open-source 'SimpleTracker' [Tinevez, J.-Y. et al. 2017].

        Args:
            localizations (np.ndarray): The sub_wavelength localization matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple which containes he structured arrays tracking information with interpolated tracks at index 0 and raw tracks at index 1.

            The different fields for interpolated tracks are:
                - pos: The interpolated sub-wavelength position of the microbubble [mm].
                - speed: The interpolated speed of the microbubble [mm/s]
                - time: The interpolated time [s].
                - track_ind: The index of the track.

            The different fields for raw tracks are:
                - pos: The interpolated sub-wavelength position of the microbubble [dimension].
                - time: The index frame.
                - track_ind: The index of the track.
        """
        start_time = time.time()
        pitch = np.mean(self.scale[:3])
        logger.debug(f"Average voxel pitch {pitch} ({self.scale[:3]})")

        # Convert localizations from pixel to [mm], frame to time in second.
        localizations["pos"] = localizations["pos"] * self.scale[:3] + self.origin

        # Tracking
        logger.debug(f"Start SimpleTracker on {len(localizations)}")
        max_linking_distance_dist = self.max_linking_distance / self.res * pitch
        logger.debug(
            f"max_linking_distance_dist: {max_linking_distance_dist} (~{max_linking_distance_dist/pitch} voxel)"
        )
        tracked_loc = SimpleTracker(
            data=localizations,
            max_linking_dist=max_linking_distance_dist,
            max_gap_closing=self.max_gap_closing,
            min_track_len=self.min_length,
        )
        # Output from SimpleTracker: ['pos', 'frame_no', 'track_no']
        track_ids = np.unique(tracked_loc["track_no"])
        # Remove localizations which are not associated to tracks.
        track_ids = track_ids[track_ids >= 0]

        # Init list for structured arrays. Collect tracks indexes.
        raw_tracks = np.zeros(
            0,
            dtype=[
                ("pos", float, 3),
                ("time", float),
                ("track_ind", int),
            ],
        )
        interp_tracks = np.zeros(
            0,
            dtype=[
                ("pos", float, 3),
                ("velocity", float, 3),
                ("time", float),
                ("track_ind", int),
            ],
        )

        logger.debug(f"{len(track_ids)} tracks found.")
        for id in track_ids:
            [raw_track, interp_track] = clean_and_interpolate_track(
                pos=tracked_loc["pos"][tracked_loc["track_no"] == id],
                index_frames=tracked_loc["frame_no"][tracked_loc["track_no"] == id],
                interp_factor=1 / self.res,
                scale=self.scale,
                track_id=id,
            )
            raw_tracks = np.append(
                raw_tracks, np.array(raw_track, dtype=raw_tracks.dtype)
            )
            interp_tracks = np.append(
                interp_tracks, np.array(interp_track, dtype=interp_tracks.dtype)
            )
        end_time = time.time()
        logger.debug(f"[create_tracks] Time taken: {end_time-start_time} sec")
        return [interp_tracks, raw_tracks]


def get_intensity_matrix(iq, fwhm, type_name) -> np.ndarray:
    """Get the intensity of voxel when a local maxima is located.

    Args:
        iq (np.ndarray): The filtered IQ data.
        fwhm (np.ndarray): Size of the PSF for localization.
        type_name (str, optional): The type of the IQ matrix. Defaults to "float".

    Returns:
        np.ndarray: An intensity matrix of size of iq. Value of the intensity of each voxel when a local maxima is found.
    """
    iq_reduced = np.zeros_like(iq, dtype=type_name)
    half_fwhm = np.ceil(1 + fwhm / 2).astype(int)
    iq_reduced[
        half_fwhm[0] : -half_fwhm[0],
        half_fwhm[1] : -half_fwhm[1],
        half_fwhm[2] : -half_fwhm[2],
        :,
    ] = iq[
        half_fwhm[0] : -half_fwhm[0],
        half_fwhm[1] : -half_fwhm[1],
        half_fwhm[2] : -half_fwhm[2],
        :,
    ].copy()

    # Concatenate height and nb frames to get a 3D matrix
    iq_reduced = np.transpose(iq_reduced, (0, 3, 1, 2))
    r_shape = iq_reduced.shape
    iq_reduced = np.reshape(iq_reduced, (-1, r_shape[-2], r_shape[-1]), order="F")

    # Find IMREGIONALMAX the 3D matrix.
    mask = maximum_filter(
        iq_reduced, footprint=np.ones((3, 3, 3)), mode="constant", cval=-1
    )
    # 0 value from IQ should not be considered as local maxima. To avoid this, change values of mask_3D from 0 to -1.
    # With this, when IQ_3D = 0, the mask of local maxima will not be set to True.
    mask[mask == 0] = -1
    # With values at -1, 0 will not be considered as a local maxima of IQ_3D.
    mask = mask == iq_reduced

    # Restore the 4D matrix
    mask = np.reshape(mask, r_shape, order="F")
    mask = np.transpose(mask, (0, 2, 3, 1))
    return mask, mask * iq


def get_local_contrast(
    iq: np.ndarray,
    mask: np.ndarray,
    intensity_matrix: np.ndarray,
    min_snr: float,
    patch_size: np.array,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieves the local contrast of local maxima to detect microbubbles.

            Args:
                iq (np.ndarray): The filtered IQ data.
                mask (np.ndarray): The binary mask where local maxima is found.
                intensity_matrix (np.ndarray): The associated value of local maxima.
                min_snr (float): min SNR to keep
                patch_size (np.ndarray): size of patch for SNR computation

    Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following elements:

                - local_contrast_matrix (np.ndarray): A 4D array representing the computed local contrast values for the input data, when a microbubble is detected and kept based on local contrast threshold.
                - final_mask (np.ndarray): A binary mask array where each element indicates the presence (`1`) or absence (`0`) of microbubbles based on the local contrast threshold.
                - linear_ind_mb_detection (np.ndarray): A 1D array of linear indices representing the positions of detected microbubbles in the flattened data structure.
                - index_frames (np.ndarray): A 1D array indicating the indices of the frames (or time steps) in which microbubbles are detected.
    """

    # Intensity to SNR.
    mat_conv = np.ones(patch_size)
    tt = np.arange(1, mat_conv.shape[0] + 1) - (mat_conv.shape[0] + 1) / 2
    [meshz, meshx, meshy] = np.meshgrid(tt, tt, tt)
    meshr = np.sqrt(meshz**2 + meshx**2 + meshy**2)
    mat_conv[meshr < np.sqrt(3) + 0.01] = 0.2  # 1rst neighbours voxels.
    mat_conv[meshr < 1] = 0  # center voxel.
    mat_conv = mat_conv / np.sum(mat_conv)
    mat_conv = np.expand_dims(mat_conv, axis=-1)

    # Compute local contrast
    clutter = iq - intensity_matrix
    clutter = np.array(convolve(clutter, mat_conv, mode="same"))

    local_contrast = intensity_matrix / (clutter)
    local_contrast[~mask] = np.nan
    local_contrast = 20 * np.log10(local_contrast)
    logger.trace(f"Apply min SNR threshold at {min_snr}dB")
    local_contrast[local_contrast < min_snr] = 0

    final_mask = local_contrast > 0
    final_mask[np.isnan(final_mask)] = 0
    final_mask = (final_mask > 0) * intensity_matrix

    final_mask_flatten_F = np.ndarray.flatten(final_mask, order="F")
    linear_ind_mb_detection = np.nonzero(final_mask_flatten_F)[0]
    logger.debug(f"{len(linear_ind_mb_detection)} maxima found")

    _, _, _, index_frames = np.unravel_index(
        linear_ind_mb_detection, final_mask.shape, order="F"
    )
    return local_contrast, final_mask, linear_ind_mb_detection, index_frames


def get_curvilinear_abscissa(m: np.ndarray, axis: int = 0):
    m = np.diff(m, axis=axis)
    d_ca = np.linalg.norm(m, ord=2, axis=1 - axis)
    ca = np.concatenate(([0], np.cumsum(d_ca)))
    return ca


def clean_and_interpolate_track(
    pos: np.array,
    scale: np.array,
    index_frames: np.array,
    interp_factor: float,
    smooth_window: int = 5,
    track_id: int = 0,
):
    """
    Clean and interpolation function to process raw microbubbles track. Perform smoothing and interpolation

    Args:
        pos (np.ndarray) [Nx3]: raw microbubble's positions.
        scale (np.ndarray) [4]: dimension scale [space, space, space, time].
        index_frames (np.ndarray) [Nx1]: frame index.
        interp_factor (float): sub scale interpolation.
        smooth_window (int): smoothing window on raw positions.
        track_id (int, optional): index of the track.

    Returns:
        list of raw data [(position, frame_index, track_id)...]
        list of interpolated data [(position, velocity, frame_index, track_id)...]
    """

    # Smooth tracks (pos : [N x 3])
    pos = smooth(pos, window=smooth_window)

    # Calculate curvilinear abscissa along track
    ca = get_curvilinear_abscissa(pos)
    ca_interp = np.arange(ca[0], ca[-1], interp_factor * np.min(scale[:3]))

    # Calculate interpolated positions and smoothing
    pos_i = np.zeros([len(ca_interp), pos.shape[1]], dtype=pos.dtype)
    for i in range(pos_i.shape[1]):
        pos_i[:, i] = np.interp(ca_interp, ca, pos[:, i])
    pos_i = smooth(pos_i, scale[i] * 2)

    # Calculate curvilinear abscissa along interpolated track
    ca_i = get_curvilinear_abscissa(pos_i)

    tl = index_frames * scale[-1]  # Track timeline
    tl_i = np.interp(ca_i / ca_i[-1], ca / ca[-1], tl)
    dt_line = np.diff(tl_i)

    # Calculate interpolated velocities in [distance/s]
    vel = np.diff(pos_i, axis=0) / np.expand_dims(dt_line, 1)
    vel = np.row_stack([vel[0, :], vel])

    raw_list = []
    for i in range(pos.shape[0]):
        raw_list.append(
            (
                pos[i, :].T,
                index_frames[i],
                track_id,
            )
        )
    interp_list = []
    for i in range(pos_i.shape[0]):
        interp_list.append(
            (
                pos_i[i, :].T,
                vel[i, :].T,
                tl_i[i],
                track_id,
            )
        )
    return [raw_list, interp_list]
