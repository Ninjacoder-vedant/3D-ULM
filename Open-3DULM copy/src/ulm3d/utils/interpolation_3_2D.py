"""
This file is used to run 3x2D RBF interpolation algorithmon complex IQ data
"""

import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger


def multiquadric_kernel(dist_matrix, epsilon):
    """
    The Multiquadric (MQ) function: phi(r) = sqrt(1 + (epsilon * r)^2)
    """
    return np.sqrt(1 + (epsilon * dist_matrix)**2)

def precompute_rbf_matrices(shape_2d, known_t_indices, epsilon):
    """
    shape_2d: (dim_spatial, dim_time_total)
    known_t_indices: list of time indices that have data
    """
    n_space, n_time = shape_2d
    
    # 1. Generate normalized coordinate grids (0 to 1)
    # This prevents 'Time' or 'Space' units from dominating the distance calculation
    x_coords = np.linspace(0, 1, n_space)
    t_coords = np.linspace(0, 1, n_time)
    
    # Create meshgrid of coordinates (Space, Time)
    T_grid, X_grid = np.meshgrid(t_coords, x_coords)
    
    # Flatten coordinates to lists of points (N_points, 2)
    all_points = np.column_stack([X_grid.ravel(), T_grid.ravel()])
    
    # 2. Separate into Known and Unknown coordinate sets
    # Create a mask for columns that are known
    mask = np.zeros(n_time, dtype=bool)
    mask[known_t_indices] = True
    
    # Expand mask to the full 2D grid
    mask_2d = np.tile(mask, (n_space, 1))
    
    known_points = all_points[mask_2d.ravel()]
    unknown_points = all_points[~mask_2d.ravel()]

    logger.debug("Known points: ",len(known_points))
    logger.debug("Unknown points: ",len(unknown_points))

    # 3. Build Distance Matrices & Kernel Matrices
    # Distances between Known points (Size: N_known x N_known)
    dist_known = cdist(known_points, known_points, metric='euclidean')
    Phi = multiquadric_kernel(dist_known, epsilon)
    
    # Distances between Unknown points and Known points (Size: N_unknown x N_known)
    dist_new = cdist(unknown_points, known_points, metric='euclidean')
    Phi_new = multiquadric_kernel(dist_new, epsilon)
    
    # 4. Compute Inverse of Phi (Regularization helps stability)
    # Adding a tiny jitter (1e-6) to diagonal for numerical stability (regularization)
    Phi_inv = np.linalg.inv(Phi + np.eye(Phi.shape[0]) * 1e-6)
    
    # The reconstruction matrix: Maps Known Values -> Unknown Values
    # Formula: F_rec = Phi_new * (Phi^-1 * F_known)
    # We combine matrices: W_interp = Phi_new * Phi^-1
    W_interp = np.dot(Phi_new,Phi_inv)
    
    return W_interp, mask_2d


def apply_rbf_2d(slice_data, W_interp, mask_2d):
    """
    Applies the precomputed RBF weights to a single data slice.
    """
    # Flatten the slice
    flat_data = slice_data.ravel()
    
    # Extract known values
    known_values = flat_data[mask_2d.ravel()]
    
    # Predict unknown values
    # Operation: [N_unknown x N_known] dot [N_known x 1] = [N_unknown x 1]
    predicted_values = np.dot(W_interp,known_values)
    
    # Fill the original array
    reconstructed_slice = slice_data.copy()
    reconstructed_slice.ravel()[~mask_2d.ravel()] = predicted_values
    
    return reconstructed_slice

def run_3x2d_upsampling_4D(iq_data, ups_factor=10, epsilon=10000.0, chunk_size=32, time_block=200):
    """
    Optimized 3x2D Algorithm with Spatial AND Temporal Chunking.
    - Spatial Chunking: Prevents RAM overflow.
    - Temporal Chunking: Fits matrix in CPU Cache (10x Speedup).
    """
    Z, X, Y, T_orig = iq_data.shape
    T_new = T_orig * ups_factor
    
    logger.trace(f"Processing Volume {Z}x{X}x{Y} | T: {T_orig}->{T_new}")
    logger.trace(f"Optimization: Spatial Chunk={chunk_size}, Time Block={time_block}")

    # 1. Allocate Accumulator (~4.6 GB)
    final_accumulator = np.zeros((Z, X, Y, T_new), dtype=np.complex64)
    
    # 2. Define Time Blocks
    # We split T_new (2000) into blocks of 'time_block' (e.g., 200)
    # The pattern of known frames is identical in each block!
    # Block 1: 0-200, Block 2: 200-400...
    t_starts = range(0, T_new, time_block)
    
    # 3. Pre-compute Matrices for ONE Time Block
    # Since every block has the same structure (every 10th frame known),
    # we calculate the matrix ONCE and reuse it for all time blocks.
    
    # Local indices relative to the block (e.g., 0, 10, 20... inside a 200 window)
    local_known_indices = np.arange(0, time_block, ups_factor)
    
    # Standard Matrix (32 x 200) -> Tiny! (~3 MB)
    print("Pre-calculating Reusable Time-Block Matrices...")
    W_std, mask_std = precompute_rbf_matrices((chunk_size, time_block), local_known_indices, epsilon)
    edge_cache = {} # To store odd-sized spatial chunks

    # --- OUTER LOOP: TIME BLOCKS ---
    for t_idx, t_start in enumerate(t_starts):
        t_end = min(t_start + time_block, T_new)
        current_t_len = t_end - t_start
        
        # Determine Input Indices corresponding to this Output Block
        # e.g., Output 0-200 corresponds to Input 0-20
        t_in_start = t_start // ups_factor
        t_in_end = t_end // ups_factor
        
        # Handle Edge Case: Last block might be shorter than 'time_block'
        if current_t_len != time_block:
            # Recalculate matrix just for the last piece
            local_indices = np.arange(0, current_t_len, ups_factor)
            W_block_std, mask_block_std = precompute_rbf_matrices((chunk_size, current_t_len), local_indices, epsilon)
            curr_edge_cache = {} # Local cache for this weird block
        else:
            # Use the cached standard matrices
            W_block_std, mask_block_std = W_std, mask_std
            curr_edge_cache = edge_cache

        # Helper to extract tiny strips from the specific time window
        def get_strip(dim_type, indices):
            # indices = [z, y, start, end] or similar
            if dim_type == 'xt':
                z, y, s, e = indices
                raw = iq_data[z, s:e, y, t_in_start:t_in_end]
            elif dim_type == 'yt':
                z, x, s, e = indices
                raw = iq_data[z, x, s:e, t_in_start:t_in_end]
            elif dim_type == 'zt':
                y, x, s, e = indices
                raw = iq_data[s:e, x, y, t_in_start:t_in_end]
            
            # Map to upsampled grid
            strip = np.zeros((indices[3]-indices[2], current_t_len), dtype=np.complex64)
            # Use local known indices relevant to this block
            valid_idx = np.arange(0, current_t_len, ups_factor)
            strip[:, valid_idx] = raw
            return strip

        # --- PASS 1: X-T Plane (For this time block) ---
        for z in range(Z):
            for y in range(Y):
                for x_start in range(0, X, chunk_size):
                    x_end = min(x_start + chunk_size, X)
                    w = x_end - x_start
                    
                    W, M = (W_block_std, mask_block_std) if w == chunk_size else \
                           get_edge_matrix(w, current_t_len, np.arange(0, current_t_len, ups_factor), epsilon, curr_edge_cache)
                    
                    strip = get_strip('xt', [z, y, x_start, x_end])
                    rec = apply_rbf_2d(strip.real, W, M) + 1j * apply_rbf_2d(strip.imag, W, M)
                    
                    # Accumulate directly to final array
                    final_accumulator[z, x_start:x_end, y, t_start:t_end] += rec

        # --- PASS 2: Y-T Plane ---
        for z in range(Z):
            for x in range(X):
                for y_start in range(0, Y, chunk_size):
                    y_end = min(y_start + chunk_size, Y)
                    w = y_end - y_start
                    
                    W, M = (W_block_std, mask_block_std) if w == chunk_size else \
                           get_edge_matrix(w, current_t_len, np.arange(0, current_t_len, ups_factor), epsilon, curr_edge_cache)
                    
                    strip = get_strip('yt', [z, x, y_start, y_end])
                    rec = apply_rbf_2d(strip.real, W, M) + 1j * apply_rbf_2d(strip.imag, W, M)
                    final_accumulator[z, x, y_start:y_end, t_start:t_end] += rec

        # --- PASS 3: Z-T Plane ---
        for y in range(Y):
            for x in range(X):
                for z_start in range(0, Z, chunk_size):
                    z_end = min(z_start + chunk_size, Z)
                    w = z_end - z_start
                    
                    W, M = (W_block_std, mask_block_std) if w == chunk_size else \
                           get_edge_matrix(w, current_t_len, np.arange(0, current_t_len, ups_factor), epsilon, curr_edge_cache)
                    
                    strip = get_strip('zt', [y, x, z_start, z_end])
                    rec = apply_rbf_2d(strip.real, W, M) + 1j * apply_rbf_2d(strip.imag, W, M)
                    final_accumulator[z_start:z_end, x, y, t_start:t_end] += rec
                    
        logger.trace(f"  Finished Time Block {t_start}-{t_end}")

    # Final Average
    # Divide in-place to save memory
    final_accumulator /= 3.0

    return final_accumulator


def get_edge_matrix(width, T, indices, eps, cache):
    """ Helper to manage the edge cache """
    if width not in cache:
        cache[width] = precompute_rbf_matrices((width, T), indices, eps)
    return cache[width]

def validate_reconstruction_4D(full_iq_data, ds, time_blk):
    """
    Robust validation for 4D data (Z, X, Y, T).
    """
    Z, X, Y, T_full = full_iq_data.shape
    
    # 1. Decimate (Create Low FPS input)
    # Keep indices: 0, ds, 2*ds... along the last axis
    decimated_data = full_iq_data[..., ::ds] 
    
    # 2. Run Reconstruction
    print(f"Validating 4D with Downsample Factor {ds}...")
    reconstructed_full = run_3x2d_upsampling_4D(decimated_data, ups_factor=ds, time_block = time_blk)
    
    # 3. Robust Comparison (Fixing the Shape Error)
    T_rec = reconstructed_full.shape[3]
    min_T = min(T_full, T_rec)
    
    # Crop both to the same timeline
    truth_cropped = full_iq_data[..., :min_T]
    pred_cropped = reconstructed_full[..., :min_T]
    
    # 4. Identify WHICH frames were interpolated
    known_indices = np.arange(0, min_T, ds)
    
    # Create a boolean mask of MISSING frames
    mask_missing = np.ones(min_T, dtype=bool)
    mask_missing[known_indices] = False
    
    # 5. Extract ONLY the missing frames for comparison
    # FIX: Use Ellipsis (...) to apply mask to the LAST dimension (Time)
    ground_truth_frames = truth_cropped[..., mask_missing]
    predicted_frames = pred_cropped[..., mask_missing]
    
    # 6. Calculate Metrics
    # NRMSE
    error_diff = predicted_frames - ground_truth_frames
    mse = np.mean(np.abs(error_diff)**2)
    rmse = np.sqrt(mse)
    
    data_range = np.max(np.abs(ground_truth_frames)) - np.min(np.abs(ground_truth_frames))
    nrmse = rmse / data_range if data_range != 0 else 0
    
    # Correlation
    flat_true = np.abs(ground_truth_frames).ravel()
    flat_pred = np.abs(predicted_frames).ravel()
    correlation = np.corrcoef(flat_true, flat_pred)[0, 1]
    
    print(f"--- Validation Results (ds={ds}) ---")
    # Note: The time dimension is now the last axis of the sliced array
    print(f"Missing Frames compared: {ground_truth_frames.shape[-1]}")
    print(f"NRMSE Error: {nrmse:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    return ground_truth_frames, predicted_frames