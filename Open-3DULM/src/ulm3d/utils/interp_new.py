"""
This file is used to run 3x2D RBF interpolation algorithm on complex IQ data
Optimized for PyTorch (Pure GPU Pipeline)
"""

import torch
import numpy as np # Only used for initial random data generation if needed
from loguru import logger
import sys

# --- CHECK GPU ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("No GPU detected. Falling back to CPU (Performance will be low).")
    device = torch.device("cpu")

def euclidean_dist_matrix_torch(A, B):
    """
    Computes Euclidean distance matrix between A and B using broadcasting.
    A: (N, D), B: (M, D) -> Result: (N, M)
    """
    # Using Torch broadcasting
    # A[:, None, :] shape: (N, 1, D)
    # B[None, :, :] shape: (1, M, D)
    diff = A[:, None, :] - B[None, :, :]
    return torch.sqrt(torch.sum(diff**2, dim=-1))

def multiquadric_kernel_torch(dist_matrix, epsilon):
    """ The Multiquadric (MQ) function """
    return torch.sqrt(1 + (epsilon * dist_matrix)**2)

def precompute_rbf_matrices_torch(shape_2d, known_t_indices, epsilon):
    """
    Generates RBF weights entirely on the device.
    shape_2d: (dim_spatial, dim_time_total)
    """
    n_space, n_time = shape_2d
    
    # 1. Generate normalized coordinate grids (0 to 1)
    x_coords = torch.linspace(0, 1, n_space, device=device)
    t_coords = torch.linspace(0, 1, n_time, device=device)
    
    # torch.meshgrid defaults to 'ij' indexing usually, we force 'xy' to match numpy behavior
    T_grid, X_grid = torch.meshgrid(t_coords, x_coords, indexing='xy')
    
    # Flatten and stack: equivalent to np.column_stack
    all_points = torch.stack([X_grid.ravel(), T_grid.ravel()], dim=1)
    
    # 2. Separate into Known and Unknown coordinate sets
    mask = torch.zeros(n_time, dtype=torch.bool, device=device)
    mask[known_t_indices] = True
    
    # Tile the mask: (n_space, n_time)
    mask_2d = mask.tile((n_space, 1))
    
    mask_flat = mask_2d.ravel()
    known_points = all_points[mask_flat]
    unknown_points = all_points[~mask_flat]

    # 3. Build Matrices
    dist_known = euclidean_dist_matrix_torch(known_points, known_points)
    Phi = multiquadric_kernel_torch(dist_known, epsilon)
    
    dist_new = euclidean_dist_matrix_torch(unknown_points, known_points)
    Phi_new = multiquadric_kernel_torch(dist_new, epsilon)
    
    # 4. Compute Inverse (Regularized)
    # Add jitter to diagonal for stability
    jitter = torch.eye(Phi.shape[0], device=device) * 1e-6
    Phi_inv = torch.linalg.inv(Phi + jitter)
    
    W_interp = torch.matmul(Phi_new, Phi_inv)
    
    return W_interp, mask_2d

def apply_rbf_2d_torch(slice_data, W_interp, mask_2d):
    """ Applies RBF weights to a single data slice on GPU. """
    if W_interp is None: return slice_data

    flat_data = slice_data.reshape(-1)
    known_values = flat_data[mask_2d.reshape(-1)]
    
    predicted_values = torch.matmul(W_interp, known_values)
    
    # Clone to avoid in-place modification errors if gradients were tracked (not here, but good practice)
    reconstructed_slice = slice_data.clone()
    
    # We need to flatten, assign, and reshape back, 
    # or rely on the fact that flat_data shares memory if it was a view.
    # Here we operate on the flat view of the clone.
    flat_rec = reconstructed_slice.view(-1)
    
    # Invert mask to find holes
    mask_flat = mask_2d.reshape(-1)
    flat_rec[~mask_flat] = predicted_values
    
    return reconstructed_slice

# Helper defined outside to be accessible
def get_edge_matrix_torch(width, T, indices, eps, cache):
    """ Helper to manage the edge cache for odd-sized chunks """
    key = (width, T)
    if key not in cache:
        cache[key] = precompute_rbf_matrices_torch((width, T), indices, eps)
    return cache[key]

def run_3x2d_upsampling_4D_torch(iq_data, ups_factor=10, epsilon=10000.0, chunk_size=32, time_block=200):
    """
    Optimized 3x2D Algorithm entirely on GPU.
    iq_data: Should be a torch tensor (complex64) on GPU. 
             If passed as numpy, it will be moved to GPU.
    """
    # Ensure Input is Tensor on Device
    if not torch.is_tensor(iq_data):
        iq_data = torch.from_numpy(iq_data)
    
    if iq_data.device != device:
        iq_data = iq_data.to(device)

    Z, X, Y, T_orig = iq_data.shape
    T_new = T_orig * ups_factor
    
    logger.info(f"Processing on [{device}] | Vol: {Z}x{X}x{Y} | T: {T_orig}->{T_new}")
    
    # 1. Allocate Accumulator on GPU
    # WARNING: This tensor can be very large. Ensure VRAM > (Z*X*Y*T_new * 8 bytes).
    # If OOM occurs, you must implement CPU offloading here.
    try:
        final_accumulator = torch.zeros((Z, X, Y, T_new), dtype=torch.complex64, device=device)
    except RuntimeError as e:
        logger.error("OOM Error allocating accumulator. Try reducing batch size or using CPU offloading.")
        raise e
    
    t_starts = range(0, T_new, time_block)
    
    # Local indices for a standard block
    local_known_indices_std = torch.arange(0, time_block, ups_factor, device=device)

    # Pre-compute Standard Matrix
    print(f"[{device}] Pre-calculating Reusable Time-Block Matrices...")
    W_std, mask_std = precompute_rbf_matrices_torch(
        (chunk_size, time_block), local_known_indices_std, epsilon
    )
    
    edge_cache = {} # Cache for edge cases

    # --- OUTER LOOP: TIME BLOCKS ---
    for t_idx, t_start in enumerate(t_starts):
        t_end = min(t_start + time_block, T_new)
        current_t_len = t_end - t_start
        t_in_start = t_start // ups_factor
        t_in_end = t_end // ups_factor
        
        # Prepare Indices for this specific time block
        local_known_indices_curr = torch.arange(0, current_t_len, ups_factor, device=device)

        # Helper to select correct matrix
        def get_matrix_for_strip(width):
            # If dimensions match standard, use standard
            if width == chunk_size and current_t_len == time_block:
                return W_std, mask_std
            else:
                # Fetch from cache (handles both time-edges and space-edges)
                return get_edge_matrix_torch(width, current_t_len, local_known_indices_curr, epsilon, edge_cache)

        # --- PROCESS STRIP HELPER ---
        def process_strip(dim_type, indices, W_matrix, Mask_matrix):
            # 1. Extract Raw Data (Direct Slice from GPU Tensor)
            if dim_type == 'xt':
                z, y, s, e = indices
                raw_device = iq_data[z, s:e, y, t_in_start:t_in_end]
            elif dim_type == 'yt':
                z, x, s, e = indices
                raw_device = iq_data[z, x, s:e, t_in_start:t_in_end]
            elif dim_type == 'zt':
                y, x, s, e = indices
                raw_device = iq_data[s:e, x, y, t_in_start:t_in_end]
            
            # 2. Prepare Strip container on GPU
            strip_shape = (indices[3]-indices[2], current_t_len)
            strip_device = torch.zeros(strip_shape, dtype=torch.complex64, device=device)
            
            # Map raw data to the upsampled grid
            valid_idx = torch.arange(0, current_t_len, ups_factor, device=device)
            strip_device[:, valid_idx] = raw_device
            
            # 3. Compute (Split Real/Imag for RBF math)
            rec_real = apply_rbf_2d_torch(strip_device.real, W_matrix, Mask_matrix)
            rec_imag = apply_rbf_2d_torch(strip_device.imag, W_matrix, Mask_matrix)
            
            # Combine back to complex
            return torch.complex(rec_real, rec_imag)

        # --- PASS 1: X-T Plane ---
        for z in range(Z):
            for y in range(Y):
                for x_start in range(0, X, chunk_size):
                    x_end = min(x_start + chunk_size, X)
                    w = x_end - x_start
                    
                    W, M = get_matrix_for_strip(w)
                    rec = process_strip('xt', [z, y, x_start, x_end], W, M)
                    
                    final_accumulator[z, x_start:x_end, y, t_start:t_end] += rec

        # --- PASS 2: Y-T Plane ---
        for z in range(Z):
            for x in range(X):
                for y_start in range(0, Y, chunk_size):
                    y_end = min(y_start + chunk_size, Y)
                    w = y_end - y_start
                    
                    W, M = get_matrix_for_strip(w)
                    rec = process_strip('yt', [z, x, y_start, y_end], W, M)
                    
                    final_accumulator[z, x, y_start:y_end, t_start:t_end] += rec

        # --- PASS 3: Z-T Plane ---
        for y in range(Y):
            for x in range(X):
                for z_start in range(0, Z, chunk_size):
                    z_end = min(z_start + chunk_size, Z)
                    w = z_end - z_start
                    
                    W, M = get_matrix_for_strip(w)
                    rec = process_strip('zt', [y, x, z_start, z_end], W, M)
                    
                    final_accumulator[z_start:z_end, x, y, t_start:t_end] += rec
        
        logger.trace(f"Finished Time Block {t_start}-{t_end}")

    final_accumulator /= 3.0
    return final_accumulator


def validate_reconstruction_4D_torch(full_iq_data, ds, time_blk):
    """
    Robust validation for 4D data using PyTorch.
    Calculates metrics on GPU if memory allows.
    """
    # Ensure input is on device
    if not torch.is_tensor(full_iq_data):
        full_iq_data = torch.from_numpy(full_iq_data)
    if full_iq_data.device != device:
        full_iq_data = full_iq_data.to(device)

    Z, X, Y, T_full = full_iq_data.shape
    
    # 1. Decimate (Slice on GPU)
    decimated_data = full_iq_data[..., ::ds] 
    
    # 2. Run Reconstruction (On GPU)
    print(f"Validating 4D with Downsample Factor {ds}...")
    reconstructed_full = run_3x2d_upsampling_4D_torch(decimated_data, ups_factor=ds, time_block=time_blk)
    
    # 3. Robust Comparison
    T_rec = reconstructed_full.shape[3]
    min_T = min(T_full, T_rec)
    
    truth_cropped = full_iq_data[..., :min_T]
    pred_cropped = reconstructed_full[..., :min_T]
    
    # 4. Identify Interpolated frames
    known_indices = torch.arange(0, min_T, ds, device=device)
    mask_missing = torch.ones(min_T, dtype=torch.bool, device=device)
    mask_missing[known_indices] = False
    
    # 5. Extract Missing Frames
    ground_truth_frames = truth_cropped[..., mask_missing]
    predicted_frames = pred_cropped[..., mask_missing]
    
    # 6. Metrics (Computed on GPU)
    error_diff = predicted_frames - ground_truth_frames
    
    # Mean Squared Error
    mse = torch.mean(torch.abs(error_diff)**2)
    rmse = torch.sqrt(mse)
    
    # NRMSE
    data_range = torch.max(torch.abs(ground_truth_frames)) - torch.min(torch.abs(ground_truth_frames))
    nrmse = rmse / data_range if data_range != 0 else torch.tensor(0.0)
    
    # Correlation
    # We flatten the ABSOLUTE values (magnitude) similar to the original logic
    flat_true = torch.abs(ground_truth_frames).reshape(-1)
    flat_pred = torch.abs(predicted_frames).reshape(-1)
    
    # Torch corrcoef expects stacked inputs [2, N]
    stacked = torch.stack((flat_true, flat_pred))
    correlation = torch.corrcoef(stacked)[0, 1]
    
    print(f"--- Validation Results (ds={ds}) ---")
    print(f"Missing Frames compared: {ground_truth_frames.shape[-1]}")
    print(f"NRMSE Error: {nrmse.item():.4f}")
    print(f"Correlation: {correlation.item():.4f}")
    
    return ground_truth_frames, predicted_frames