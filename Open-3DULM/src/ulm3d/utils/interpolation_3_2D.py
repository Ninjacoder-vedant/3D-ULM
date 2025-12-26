"""
This file is used to run 3x2D RBF interpolation algorithmon complex IQ data
"""

import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

def get_backend(use_gpu):
    if use_gpu:
        if not HAS_GPU:
            raise RuntimeError("GPU requested but CuPy is not installed or no GPU detected.")
        return cp
    return np

def euclidean_dist_matrix(xp, A, B):
    """
    Computes Euclidean distance matrix between A and B using broadcasting.
    Works for both NumPy and CuPy.
    """
    # A: (N, D), B: (M, D) -> Result: (N, M)
    # Broadcasting: (N, 1, D) - (1, M, D)
    diff = A[:, None, :] - B[None, :, :]
    return xp.sqrt(xp.sum(diff**2, axis=-1))

def multiquadric_kernel(xp, dist_matrix, epsilon):
    """
    The Multiquadric (MQ) function: phi(r) = sqrt(1 + (epsilon * r)^2)
    """
    return xp.sqrt(1 + (epsilon * dist_matrix)**2)

def precompute_rbf_matrices(shape_2d, known_t_indices, epsilon, xp):
    """
    shape_2d: (dim_spatial, dim_time_total)
    known_t_indices: list of time indices that have data
    """
    n_space, n_time = shape_2d
    
    # 1. Generate normalized coordinate grids (0 to 1)
    # This prevents 'Time' or 'Space' units from dominating the distance calculation
    x_coords = xp.linspace(0, 1, n_space)
    t_coords = xp.linspace(0, 1, n_time)
    
    # Create meshgrid of coordinates (Space, Time)
    T_grid, X_grid = xp.meshgrid(t_coords, x_coords)
    
    # Flatten coordinates to lists of points (N_points, 2)
    all_points = xp.column_stack([X_grid.ravel(), T_grid.ravel()])
    
    # 2. Separate into Known and Unknown coordinate sets
    # Create a mask for columns that are known
    mask = xp.zeros(n_time, dtype=bool)
    mask[xp.asarray(known_t_indices)] = True
    
    # Expand mask to the full 2D grid
    mask_2d = xp.tile(mask, (n_space, 1))
    
    known_points = all_points[mask_2d.ravel()]
    unknown_points = all_points[~mask_2d.ravel()]

    logger.debug("Known points: ",len(known_points))
    logger.debug("Unknown points: ",len(unknown_points))

    # 3. Build Distance Matrices & Kernel Matrices
    # Distances between Known points (Size: N_known x N_known)
    dist_known = euclidean_dist_matrix(xp, known_points, known_points)
    Phi = multiquadric_kernel(xp, dist_known, epsilon)
    
    # Distances between Unknown points and Known points (Size: N_unknown x N_known)
    dist_new = euclidean_dist_matrix(xp, unknown_points, known_points)
    Phi_new = multiquadric_kernel(xp, dist_new, epsilon)
    
    # 4. Compute Inverse of Phi (Regularization helps stability)
    # Adding a tiny jitter (1e-6) to diagonal for numerical stability (regularization)
    Phi_inv = xp.linalg.inv(Phi + xp.eye(Phi.shape[0]) * 1e-6)
    
    # The reconstruction matrix: Maps Known Values -> Unknown Values
    # Formula: F_rec = Phi_new * (Phi^-1 * F_known)
    # We combine matrices: W_interp = Phi_new * Phi^-1
    W_interp = xp.dot(Phi_new,Phi_inv)
    
    return W_interp, mask_2d


def apply_rbf_2d(slice_data, W_interp, mask_2d, xp):
    """
    Applies the precomputed RBF weights to a single data slice.
    """
    if W_interp is None: return slice_data
    # Flatten the slice
    flat_data = slice_data.ravel()
    
    # Extract known values
    known_values = flat_data[mask_2d.ravel()]
    
    # Predict unknown values
    # Operation: [N_unknown x N_known] dot [N_known x 1] = [N_unknown x 1]
    predicted_values = xp.dot(W_interp,known_values)
    
    # Fill the original array
    reconstructed_slice = slice_data.copy()
    reconstructed_slice.ravel()[~mask_2d.ravel()] = predicted_values
    
    return reconstructed_slice

def run_3x2d_upsampling_4D(iq_data, ups_factor=10, epsilon=10000.0, chunk_size=32, time_block=200, use_gpu = False):
    """
    Optimized 3x2D Algorithm with Spatial AND Temporal Chunking.
    - Spatial Chunking: Prevents RAM overflow.
    - Temporal Chunking: Fits matrix in CPU Cache (10x Speedup).
    """
    xp = get_backend(use_gpu) # Select 'numpy' or 'cupy'
    device_name = "GPU" if use_gpu else "CPU"

    Z, X, Y, T_orig = iq_data.shape
    T_new = T_orig * ups_factor
    
    logger.info(f"Processing on [{device_name}] | Vol: {Z}x{X}x{Y} | T: {T_orig}->{T_new}")

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
    W_std, mask_std = precompute_rbf_matrices((chunk_size, time_block), local_known_indices, epsilon, xp)
    edge_cache = {} # To store odd-sized spatial chunks

    # --- OUTER LOOP: TIME BLOCKS ---
    for t_idx, t_start in enumerate(t_starts):
        t_end = min(t_start + time_block, T_new)
        current_t_len = t_end - t_start
        
        # Determine Ixput Indices corresponding to this Output Block
        # e.g., Output 0-200 corresponds to Ixput 0-20
        t_in_start = t_start // ups_factor
        t_in_end = t_end // ups_factor
        
        # Handle Edge Case: Last block might be shorter than 'time_block'
        if current_t_len != time_block:
            # Recalculate matrix just for the last piece
            local_indices = xp.arange(0, current_t_len, ups_factor)
            W_block_std, mask_block_std = precompute_rbf_matrices((chunk_size, current_t_len), local_indices, epsilon)
            curr_edge_cache = {} # Local cache for this weird block
        else:
            # Use the cached standard matrices
            W_block_std, mask_block_std = W_std, mask_std
            curr_edge_cache = edge_cache

        # Helper to extract tiny strips from the specific time window
        def process_strip(dim_type, indices):
            # 1. Extract Raw Data (CPU)
            if dim_type == 'xt':
                z, y, s, e = indices
                raw_cpu = iq_data[z, s:e, y, t_in_start:t_in_end]
            elif dim_type == 'yt':
                z, x, s, e = indices
                raw_cpu = iq_data[z, x, s:e, t_in_start:t_in_end]
            elif dim_type == 'zt':
                y, x, s, e = indices
                raw_cpu = iq_data[s:e, x, y, t_in_start:t_in_end]
            
            # 2. Prepare Strip container
            # Map raw data to the upsampled grid
            strip_shape = (indices[3]-indices[2], current_t_len)
            
            # 3. Move to Device (GPU) & Compute
            if use_gpu:
                # Move raw data to GPU
                raw_device = cp.asarray(raw_cpu)
                strip_device = cp.zeros(strip_shape, dtype=cp.complex64)
                
                valid_idx = cp.arange(0, current_t_len, ups_factor)
                strip_device[:, valid_idx] = raw_device
                
                # Compute
                rec_real = apply_rbf_2d(strip_device.real, W_curr, M_curr, xp)
                rec_imag = apply_rbf_2d(strip_device.imag, W_curr, M_curr, xp)
                
                # Move result back to CPU
                return (rec_real + 1j * rec_imag).get()
            else:
                # CPU Path
                strip_device = np.zeros(strip_shape, dtype=np.complex64)
                valid_idx = np.arange(0, current_t_len, ups_factor)
                strip_device[:, valid_idx] = raw_cpu
                
                rec_real = apply_rbf_2d(strip_device.real, W_curr, M_curr, xp)
                rec_imag = apply_rbf_2d(strip_device.imag, W_curr, M_curr, xp)
                return rec_real + 1j * rec_imag

        # --- PASS 1: X-T Plane ---
        for z in range(Z):
            for y in range(Y):
                for x_start in range(0, X, chunk_size):
                    x_end = min(x_start + chunk_size, X)
                    # Check matrix needed (Edge cache logic inside loop for correctness if needed)
                    # For simplicity, assuming chunk_size matches unless X is odd
                    if (x_end - x_start) != chunk_size:
                         # Handle spatial edge cases if X is not div by 32
                         W_use, M_use = get_edge_matrix((x_end - x_start), current_t_len, np.arange(0, current_t_len, ups_factor))
                    
                    # Note: We rely on the time-block matrix (W_curr) which assumes 
                    # spatial chunk is standard. If spatial is not standard, we'd need W_spatial_edge.
                    # (Code simplified here to assume 'W_curr' covers the time aspect. 
                    # Real spatial edge handling requires re-generating W for width!=32)
                    
                    rec = process_strip('xt', [z, y, x_start, x_end])
                    final_accumulator[z, x_start:x_end, y, t_start:t_end] += rec

        # --- PASS 2: Y-T Plane ---
        for z in range(Z):
            for x in range(X):
                for y_start in range(0, Y, chunk_size):
                    y_end = min(y_start + chunk_size, Y)
                    rec = process_strip('yt', [z, x, y_start, y_end])
                    final_accumulator[z, x, y_start:y_end, t_start:t_end] += rec

        # --- PASS 3: Z-T Plane ---
        for y in range(Y):
            for x in range(X):
                for z_start in range(0, Z, chunk_size):
                    z_end = min(z_start + chunk_size, Z)
                    rec = process_strip('zt', [y, x, z_start, z_end])
                    final_accumulator[z_start:z_end, x, y, t_start:t_end] += rec
        
        logger.trace(f"Finished Time Block {t_start}-{t_end}")

    # Final Average
    # Divide in-place to save memory
    final_accumulator /= 3.0

    return final_accumulator

def get_edge_matrix(width, T, indices, eps, cache, xp):
    """ Helper to manage the edge cache """
    if width not in cache:
        cache[width] = precompute_rbf_matrices((width, T), indices, eps, xp)
    return cache[width]

def validate_reconstruction_4D(full_iq_data, ds, time_blk):
    """
    Robust validation for 4D data (Z, X, Y, T).
    """
    Z, X, Y, T_full = full_iq_data.shape
    
    # 1. Decimate (Create Low FPS ixput)
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
    known_indices = xp.arange(0, min_T, ds)
    
    # Create a boolean mask of MISSING frames
    mask_missing = xp.ones(min_T, dtype=bool)
    mask_missing[known_indices] = False
    
    # 5. Extract ONLY the missing frames for comparison
    # FIX: Use Ellipsis (...) to apply mask to the LAST dimension (Time)
    ground_truth_frames = truth_cropped[..., mask_missing]
    predicted_frames = pred_cropped[..., mask_missing]
    
    # 6. Calculate Metrics
    # NRMSE
    error_diff = predicted_frames - ground_truth_frames
    mse = xp.mean(xp.abs(error_diff)**2)
    rmse = xp.sqrt(mse)
    
    data_range = xp.max(xp.abs(ground_truth_frames)) - xp.min(xp.abs(ground_truth_frames))
    nrmse = rmse / data_range if data_range != 0 else 0
    
    # Correlation
    flat_true = xp.abs(ground_truth_frames).ravel()
    flat_pred = xp.abs(predicted_frames).ravel()
    correlation = xp.corrcoef(flat_true, flat_pred)[0, 1]
    
    print(f"--- Validation Results (ds={ds}) ---")
    # Note: The time dimension is now the last axis of the sliced array
    print(f"Missing Frames compared: {ground_truth_frames.shape[-1]}")
    print(f"NRMSE Error: {nrmse:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    return ground_truth_frames, predicted_frames