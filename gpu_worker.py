import pathlib
import h5py
import mat73
import numpy as np
from scipy.io import loadmat
import time
import sys
from loguru import logger

import pathlib
import h5py
import mat73
import numpy as np
from scipy.io import loadmat

def load_iq(iq_file: str, input_var_name="") -> np.ndarray:
    ext = pathlib.Path(iq_file).suffix
    if ext == ".mat":
        # Load IQ.
        try:
            iq = loadmat(iq_file, squeeze_me=True)[input_var_name]
        except NotImplementedError:
            iq = mat73.loadmat(iq_file)[input_var_name]  # If IQ is saved in 7.3.
    elif ext == ".npy":
        iq = np.load(iq_file)

    elif ext == ".npz":
        iq = np.load(iq_file)[input_var_name]

    else:
        raise NotImplementedError(
            f"Loading iq with extension {ext} is not implemented."
        )
    return iq

# --- GPU SETUP ---
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# --- 1. GPU MATH KERNELS ---
def multiquadric_kernel(xp, dist_matrix, epsilon):
    return xp.sqrt(1 + (epsilon * dist_matrix)**2)

def precompute_rbf_matrices_gpu(shape_2d, known_t_indices, epsilon):
    n_space, n_time = shape_2d
    x_coords = cp.linspace(0, 1, n_space)
    t_coords = cp.linspace(0, 1, n_time)
    T_grid, X_grid = cp.meshgrid(t_coords, x_coords)
    all_points = cp.column_stack([X_grid.ravel(), T_grid.ravel()])
    
    mask = cp.zeros(n_time, dtype=bool)
    mask[cp.asarray(known_t_indices)] = True
    mask_2d = cp.tile(mask, (n_space, 1))
    
    known_points = all_points[mask_2d.ravel()]
    unknown_points = all_points[~mask_2d.ravel()]

    diff_known = known_points[:, None, :] - known_points[None, :, :]
    dist_known = cp.sqrt(cp.sum(diff_known**2, axis=-1))
    diff_new = unknown_points[:, None, :] - known_points[None, :, :]
    dist_new = cp.sqrt(cp.sum(diff_new**2, axis=-1))

    Phi = multiquadric_kernel(cp, dist_known, epsilon)
    Phi_new = multiquadric_kernel(cp, dist_new, epsilon)
    Phi_inv = cp.linalg.inv(Phi + cp.eye(Phi.shape[0]) * 1e-6)
    
    return cp.dot(Phi_new, Phi_inv), mask_2d

def apply_rbf_gpu(slice_data, W, mask_2d):
    flat = slice_data.ravel()
    known = flat[mask_2d.ravel()]
    pred = cp.dot(W, known)
    res = slice_data.copy()
    res.ravel()[~mask_2d.ravel()] = pred
    return res

# --- 2. MAIN ALGO ---
def run_3x2d_upsampling_4D(iq_data_cpu, ups_factor=10, epsilon=50.0, chunk_size=32, time_block=200):
    if not HAS_GPU: return None, 0
    start_t = time.time()
    Z, X, Y, T_orig = iq_data_cpu.shape
    T_new = T_orig * ups_factor
    
    iq_data_gpu = cp.asarray(iq_data_cpu)
    final_acc = cp.zeros((Z, X, Y, T_new), dtype=cp.complex64)
    
    local_known_indices = np.arange(0, time_block, ups_factor)
    W_std, mask_std = precompute_rbf_matrices_gpu((chunk_size, time_block), local_known_indices, epsilon)
    edge_cache = {}

    def get_matrix(w, t):
        key = (w, t)
        if key not in edge_cache:
            loc_idx = np.arange(0, t, ups_factor)
            edge_cache[key] = precompute_rbf_matrices_gpu((chunk_size, t), loc_idx, epsilon)
        return edge_cache[key]

    t_starts = range(0, T_new, time_block)
    for t_start in t_starts:
        t_end = min(t_start + time_block, T_new)
        curr_t = t_end - t_start
        t_in_start = t_start // ups_factor
        t_in_end = t_end // ups_factor

        W, M = (W_std, mask_std) if curr_t == time_block else get_matrix(chunk_size, curr_t)

        for z in range(Z):
            for y in range(Y):
                for x_start in range(0, X, chunk_size):
                    x_end = min(x_start + chunk_size, X)
                    strip = iq_data_gpu[z, x_start:x_end, y, t_in_start:t_in_end]
                    strip_up = cp.zeros((x_end-x_start, curr_t), dtype=cp.complex64)
                    strip_up[:, ::ups_factor] = strip
                    rec = apply_rbf_gpu(strip_up.real, W, M) + 1j * apply_rbf_gpu(strip_up.imag, W, M)
                    final_acc[z, x_start:x_end, y, t_start:t_end] += rec
        
        for z in range(Z):
            for x in range(X):
                for y_start in range(0, Y, chunk_size):
                    y_end = min(y_start + chunk_size, Y)
                    strip = iq_data_gpu[z, x, y_start:y_end, t_in_start:t_in_end]
                    strip_up = cp.zeros((y_end-y_start, curr_t), dtype=cp.complex64)
                    strip_up[:, ::ups_factor] = strip
                    rec = apply_rbf_gpu(strip_up.real, W, M) + 1j * apply_rbf_gpu(strip_up.imag, W, M)
                    final_acc[z, x, y_start:y_end, t_start:t_end] += rec

        for y in range(Y):
            for x in range(X):
                for z_start in range(0, Z, chunk_size):
                    z_end = min(z_start + chunk_size, Z)
                    strip = iq_data_gpu[z_start:z_end, x, y, t_in_start:t_in_end]
                    strip_up = cp.zeros((z_end-z_start, curr_t), dtype=cp.complex64)
                    strip_up[:, ::ups_factor] = strip
                    rec = apply_rbf_gpu(strip_up.real, W, M) + 1j * apply_rbf_gpu(strip_up.imag, W, M)
                    final_acc[z_start:z_end, x, y, t_start:t_end] += rec

    final_acc /= 3.0
    result_cpu = final_acc.get()
    
    del final_acc
    del iq_data_gpu
    del W_std
    cp.get_default_memory_pool().free_all_blocks()
    
    return result_cpu, time.time() - start_t

def worker_task(file_idx):
    try:
        # Load your data here
        file_path = f"mat/IQ{file_idx+1:03d}.mat"
        iq_mock = load_iq(file_path,"IQ")

        # Run GPU Algo
        res, t_gpu = run_3x2d_upsampling_4D(iq_mock, ups_factor=10)
    
        del iq_mock
        del res
        return f"Dataset {file_idx}: Done in {t_gpu:.2f}s"
    except Exception as e:
        return f"Error {file_idx}: {e}"
