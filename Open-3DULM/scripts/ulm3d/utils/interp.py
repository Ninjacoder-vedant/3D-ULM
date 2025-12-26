import torch
import numpy as np
from loguru import logger
import sys

# --- CHECK GPU ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("No GPU detected.")
    device = torch.device("cpu")

# --- RBF KERNEL FUNCTIONS (Unchanged logic, just helpers) ---
def euclidean_dist_matrix_torch(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return torch.sqrt(torch.sum(diff**2, dim=-1))

def multiquadric_kernel_torch(dist_matrix, epsilon):
    return torch.sqrt(1 + (epsilon * dist_matrix)**2)

def precompute_rbf_matrices_torch(shape_2d, known_t_indices, epsilon):
    """
    Generates RBF weights.
    shape_2d: (dim_spatial, dim_time_total)
    """
    n_space, n_time = shape_2d
    
    # Grid generation
    x_coords = torch.linspace(0, 1, n_space, device=device)
    t_coords = torch.linspace(0, 1, n_time, device=device)
    T_grid, X_grid = torch.meshgrid(t_coords, x_coords, indexing='xy')
    all_points = torch.stack([X_grid.ravel(), T_grid.ravel()], dim=1)
    
    # Masking
    mask = torch.zeros(n_time, dtype=torch.bool, device=device)
    mask[known_t_indices] = True
    mask_2d = mask.tile((n_space, 1))
    
    mask_flat = mask_2d.ravel()
    known_points = all_points[mask_flat]
    unknown_points = all_points[~mask_flat]

    # Matrices
    dist_known = euclidean_dist_matrix_torch(known_points, known_points)
    Phi = multiquadric_kernel_torch(dist_known, epsilon)
    
    dist_new = euclidean_dist_matrix_torch(unknown_points, known_points)
    Phi_new = multiquadric_kernel_torch(dist_new, epsilon)
    
    # Inverse with jitter
    jitter = torch.eye(Phi.shape[0], device=device) * 1e-6
    Phi_inv = torch.linalg.inv(Phi + jitter)
    
    W_interp = torch.matmul(Phi_new, Phi_inv)
    return W_interp, mask_2d

def apply_vectorized_rbf(flat_data, W_interp, mask_2d):
    """
    Applies RBF to a massively flattened batch.
    flat_data: (Big_Batch, Spatial * Time)
    W_interp: (N_unknown, N_known)
    """
    if W_interp is None: return flat_data

    # Shapes
    B_total = flat_data.shape[0]
    mask_flat = mask_2d.reshape(-1) # (Spatial*Time)
    
    # 1. Extract Known Values
    # flat_data is (B, N_points). mask_flat is (N_points).
    # Boolean indexing selects columns: result (B, N_known)
    known_values = flat_data[:, mask_flat]
    
    # 2. Matrix Multiplication
    # (B, N_known) @ (N_unknown, N_known).T -> (B, N_unknown)
    predicted_values = torch.matmul(known_values, W_interp.T)
    
    # 3. Reconstruction
    # Clone to avoid overwriting input if needed (safer for gradients/views)
    out_data = flat_data.clone()
    
    # Assign back to unknown slots
    # We cannot use simple boolean indexing for assignment on the flattened 2D tensor easily 
    # without advanced indexing. 
    # out_data[:, ~mask_flat] = predicted_values works efficiently in PyTorch.
    out_data[:, ~mask_flat] = predicted_values
    
    return out_data

def get_cached_matrix(width, time_len, indices, eps, cache):
    key = (width, time_len)
    if key not in cache:
        cache[key] = precompute_rbf_matrices_torch((width, time_len), indices, eps)
    return cache[key]

# --- MAIN OPTIMIZED FUNCTION ---
@torch.no_grad()
def run_fast_upsampling(batched_data, ups_factor=10, epsilon=10000.0, time_block=200):
    """
    Vectorized 3x2D Upsampling.
    Removes spatial loops by treating dimensions as batch items.
    """
    # 1. Setup Data
    if not torch.is_tensor(batched_data):
        batched_data = torch.from_numpy(batched_data)
    if batched_data.dim() == 4:
        batched_data = batched_data.unsqueeze(0)
    
    batched_data = batched_data.to(device)
    
    Batch, Z, X, Y, T_orig = batched_data.shape
    T_new = T_orig * ups_factor
    
    logger.info(f"Fast Processing | Shape: {Batch}x{Z}x{X}x{Y} | T: {T_orig}->{T_new}")

    # 2. Allocate Accumulator (10GB for 2000 frames)
    # 
    # final_accumulator = torch.zeros((Batch, Z, X, Y, T_new), dtype=torch.complex64, device=device)
    final_accumulator_cpu = torch.zeros(
        (Batch, Z, X, Y, T_new), 
        dtype=torch.complex64, 
        device='cpu', 
        pin_memory=True 
    )
    
    # 3. Setup Time Blocks
    t_starts = range(0, T_new, time_block)
    
    # Cache for RBF matrices (X, Y, Z might differ in size)
    matrix_cache = {}
    
    # --- OUTER LOOP: TIME BLOCKS ONLY ---
    for t_start in t_starts:
        t_end = min(t_start + time_block, T_new)
        curr_t_len = t_end - t_start
        t_in_start = t_start // ups_factor
        t_in_end = t_end // ups_factor
        
        # Shape: (B, Z, X, Y, curr_t_len) - Much smaller than full volume
        block_accumulator_gpu = torch.zeros(
            (Batch, Z, X, Y, curr_t_len), 
            dtype=torch.complex64, 
            device=device
        )
        
        # Extract small time slice from input: (B, Z, X, Y, T_small)
        # We need to handle edge case where input slice might be smaller at the very end
        t_in_slice = batched_data[..., t_in_start:t_in_end]
        
        # Calculate indices for this block
        local_indices = torch.arange(0, curr_t_len, ups_factor, device=device)

        # --- PASS 1: X-T Plane ---
        # We want to interpolate X-T. Treat (B, Z, Y) as batch.
        # Permute to: (B, Z, Y, X, T)
        # Reshape to: (B*Z*Y, X*T)
        input_view_xt = t_in_slice.permute(0, 1, 3, 2, 4).reshape(-1, X * (t_in_end - t_in_start))
        
        # Check cache / Precompute Matrix for dimension X
        W, M = get_cached_matrix(X, curr_t_len, local_indices, epsilon, matrix_cache)
        
        # Prepare Sparse Grid (B_total, X * T_large)
        sparse_grid = torch.zeros((input_view_xt.shape[0], X * curr_t_len), dtype=torch.complex64, device=device)
        
        # Map input to sparse grid
        # We need to construct a mask mapping input pixels to output pixels
        # M is (X, T_new). M.ravel() has True at known locations.
        # We just copy directly using the known mask logic
        valid_mask_flat = M.reshape(-1) # (X * T_large)
        sparse_grid[:, valid_mask_flat] = input_view_xt
        
        # Run RBF
        rec_real = apply_vectorized_rbf(sparse_grid.real, W, M)
        rec_imag = apply_vectorized_rbf(sparse_grid.imag, W, M)
        rec_xt = torch.complex(rec_real, rec_imag)
        
        # Reshape and Add to Accumulator
        # Shape back to: (B, Z, Y, X, T) -> Permute back to (B, Z, X, Y, T)
        rec_xt = rec_xt.reshape(Batch, Z, Y, X, curr_t_len).permute(0, 1, 3, 2, 4)
        block_accumulator_gpu += rec_xt
        del rec_xt, input_view_xt, sparse_grid, rec_real, rec_imag

        # --- PASS 2: Y-T Plane ---
        # Active: Y, T. Batch: B, Z, X.
        # Permute to: (B, Z, X, Y, T) -> This is natural order!
        # Reshape to: (B*Z*X, Y*T)
        input_view_yt = t_in_slice.reshape(-1, Y * (t_in_end - t_in_start))
        
        W, M = get_cached_matrix(Y, curr_t_len, local_indices, epsilon, matrix_cache)
        
        sparse_grid = torch.zeros((input_view_yt.shape[0], Y * curr_t_len), dtype=torch.complex64, device=device)
        valid_mask_flat = M.reshape(-1)
        sparse_grid[:, valid_mask_flat] = input_view_yt
        
        rec_real = apply_vectorized_rbf(sparse_grid.real, W, M)
        rec_imag = apply_vectorized_rbf(sparse_grid.imag, W, M)
        rec_yt = torch.complex(rec_real, rec_imag)
        
        # Reshape: (B, Z, X, Y, T) -> Natural order, no permute needed
        rec_yt = rec_yt.reshape(Batch, Z, X, Y, curr_t_len)
        block_accumulator_gpu += rec_yt
        
        del rec_yt, input_view_yt, sparse_grid, rec_real, rec_imag

        # --- PASS 3: Z-T Plane ---
        # Active: Z, T. Batch: B, X, Y.
        # Permute to: (B, X, Y, Z, T)
        input_view_zt = t_in_slice.permute(0, 2, 3, 1, 4).reshape(-1, Z * (t_in_end - t_in_start))
        
        W, M = get_cached_matrix(Z, curr_t_len, local_indices, epsilon, matrix_cache)
        
        sparse_grid = torch.zeros((input_view_zt.shape[0], Z * curr_t_len), dtype=torch.complex64, device=device)
        valid_mask_flat = M.reshape(-1)
        sparse_grid[:, valid_mask_flat] = input_view_zt
        
        rec_real = apply_vectorized_rbf(sparse_grid.real, W, M)
        rec_imag = apply_vectorized_rbf(sparse_grid.imag, W, M)
        rec_zt = torch.complex(rec_real, rec_imag)
        
        # Reshape: (B, X, Y, Z, T) -> Permute to (B, Z, X, Y, T)
        rec_zt = rec_zt.reshape(Batch, X, Y, Z, curr_t_len).permute(0, 3, 1, 2, 4)
        block_accumulator_gpu += rec_zt
        
        del rec_zt, input_view_zt, sparse_grid, rec_real, rec_imag

        # [NEW] Normalize, Offload to CPU, and Flush GPU Memory
        block_accumulator_gpu /= 3.0
        final_accumulator_cpu[..., t_start:t_end] = block_accumulator_gpu.cpu()
        
        del block_accumulator_gpu
        torch.cuda.empty_cache() # Crucial: Return freed memory to OS/Allocator
        
        logger.debug(f"Time Block {t_start}-{t_end} processed.")

    final_accumulator_cpu /= 3.0