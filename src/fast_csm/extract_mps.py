import time
import torch
import sigpy as sp

from tqdm import tqdm
from typing import Optional
from einops import rearrange, einsum
from mr_recon.utils.func import np_to_torch, torch_to_np, sp_ifft

def csm_from_espirit(ksp_cal: torch.Tensor,
                     im_size: tuple,
                     thresh: Optional[float] = 0.02,
                     kernel_width: Optional[int] = 6,
                     crp: Optional[float] = 0.95,
                     max_iter: Optional[int] = 100,
                     verbose: Optional[bool] = True) -> (torch.Tensor, torch.Tensor):
    """
    Copy of sigpy implementation of ESPIRiT calibration, but in torch:
    Martin Uecker, ... ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (ncoil, *cal_size)
    im_size : tuple
        output image size
    thresh : float
        threshold for SVD nullspace
    kernel_width : int
        width of calibration kernel
    crp : float
        output mask based on copping eignevalues
    max_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Consts
    img_ndim = ksp_cal.ndim - 1
    num_coils = ksp_cal.shape[0]
    device = ksp_cal.device

    # TODO torch this part
    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    ksp_cal_sp = torch_to_np(ksp_cal)
    dev = sp.get_device(ksp_cal_sp)
    with dev:
        mat = sp.array_to_blocks(
            ksp_cal_sp, [kernel_width] * img_ndim, [1] * img_ndim)
        mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
        mat = mat.transpose([1, 0, 2])
        mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
    mat = np_to_torch(mat)

    # Perform SVD on calibration matrix
    if verbose:
        print('Computing SVD on calibration matrix')
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]

    # Get kernels
    num_kernels = len(VH)
    kernels = VH.reshape(
        [num_kernels, num_coils] + [kernel_width] * img_ndim)

    # Get covariance matrix in image domain
    AHA = torch.zeros(im_size + (num_coils, num_coils), 
                        dtype=ksp_cal.dtype, device=device)
    for kernel in tqdm(kernels, 'Computing covariance matrix', disable=not verbose):
        img_kernel = sp_ifft(kernel, oshape=(num_coils, *im_size),
                                dim=tuple(range(-img_ndim, 0)))
        aH = rearrange(img_kernel, 'nc ... -> ... nc 1')
        a = aH.swapaxes(-1, -2).conj()
        AHA += aH @ a
    AHA *= (torch.prod(torch.tensor(im_size)).item() / kernel_width**img_ndim)
    
    # Get eigenvalues and eigenvectors
    mps, eigen_vals = power_method(AHA, num_iter=max_iter, verbose=verbose)
    
    # Phase relative to first map
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
    mps *= eigen_vals > crp

    return mps, eigen_vals

def csm_from_kernels(grappa_kernels: torch.Tensor,
                     source_vectors: torch.Tensor,
                     im_size: tuple,
                     crp: Optional[float] = 0.95,
                     num_iter: Optional[int] = 100,
                     verbose: Optional[bool] = True) -> (torch.Tensor, torch.Tensor):
    """
    Estimates coil sensitivty maps from grappa kernels

    Parameters:
    -----------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    im_size : tuple
        output image size
    crp : float
        crops based on eignevalues
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    BHB = calc_image_covariance_kernels(grappa_kernels, source_vectors, im_size, verbose)
    mps, eigen_vals = power_method(BHB, num_iter, verbose)

    # Phase relative to first map
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
    mps *= eigen_vals > crp

    return mps, eigen_vals

def calc_image_covariance_kernels(grappa_kernels: torch.Tensor,
                                  source_vectors: torch.Tensor,
                                  im_size: tuple,
                                  verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates B^HB matrix -- see writeup for details

    Parameters:
    -----------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    im_size : tuple
        output image size
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    BHB : torch.Tensor
        image covariance kernels with shape (ncoil, ncoil, *im_size)
    """

    # Consts
    device = grappa_kernels.device
    nkerns, ncoil, _, num_inputs = grappa_kernels.shape
    d = source_vectors.shape[-1]
    assert nkerns == source_vectors.shape[0]
    assert num_inputs == source_vectors.shape[1]
    assert device == source_vectors.device

    # Make cross terms
    grappa_kerns_rs = rearrange(grappa_kernels, 'N nco nci ninp -> nco nci N ninp')
    source_vectors_cross = source_vectors[:, :, None, :] - source_vectors[:, None, :, :] # nkerns ninp ninp d
    grappa_kerns_cross = einsum(grappa_kerns_rs.conj(), grappa_kerns_rs, 
                        'nc nci B ninp, nc nci2 B ninp2 -> nci nci2 B ninp ninp2')
    
    # Rescaling
    scale = (torch.prod(torch.tensor(im_size)).item()) ** 0.5 / nkerns
    grappa_kerns_rs *= scale
    grappa_kerns_cross *= scale

    # Using sigpy for adjoint nufft purposes
    grappa_kerns_cross_sp = torch_to_np(grappa_kerns_cross)
    grappa_kerns_rs_sp = torch_to_np(grappa_kerns_rs)
    source_vectors_cross_sp = torch_to_np(source_vectors_cross)
    source_vectors_sp = torch_to_np(source_vectors)
    dev = sp.get_device(grappa_kerns_cross_sp)
    xp = dev.xp

    # Build BHB matrix
    with dev:
        BHB = xp.zeros((ncoil, ncoil, *im_size), dtype=grappa_kerns_rs_sp.dtype)
        batch = False
        if batch:
            coil_batch = ncoil # TODO: make this a parameter
            for i in tqdm(range(ncoil), 'Computing Covariance Matrix', disable=not verbose):
                for j in range(0, ncoil, coil_batch):
                    c1 = j
                    c2 = min(ncoil, j + coil_batch)
                    oshape = (c2 - c1, *im_size)
                    BHB[i,c1:c2] += sp.nufft_adjoint(grappa_kerns_cross_sp[i,c1:c2], source_vectors_cross_sp, oshape, width=6)
                    BHB[i,c1:c2] += -sp.nufft_adjoint(grappa_kerns_rs_sp[i,c1:c2], -source_vectors_sp, oshape, width=6)
                    BHB[i,c1:c2] += -sp.nufft_adjoint(xp.swapaxes(grappa_kerns_rs_sp.conj(), 0, 1)[i,c1:c2], source_vectors_sp, oshape, width=6)
            # BHB[i,i] += 1
        else:
            if verbose:
                print(f'Computing Covariance Matrix: ', end='')
                start = time.perf_counter()
            oshape = (ncoil, ncoil, *im_size)
            BHB += sp.nufft_adjoint(grappa_kerns_cross_sp, source_vectors_cross_sp, oshape, width=6)
            BHB += -sp.nufft_adjoint(grappa_kerns_rs_sp, -source_vectors_sp, oshape, width=6)
            BHB += -sp.nufft_adjoint(xp.swapaxes(grappa_kerns_rs_sp.conj(), 0, 1), source_vectors_sp, oshape, width=6)
            if verbose:
                end = time.perf_counter()
                print(f'{end - start:.3f}s')
        BHB = rearrange(BHB, 'nc nci ... -> ... nc nci')
    
    return np_to_torch(BHB)
    
def power_method(M: torch.Tensor,
                 num_iter: Optional[int] = 100,
                 verbose: Optional[bool] = True) -> (torch.Tensor, torch.Tensor):
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    M : torch.Tensor
        input matrix with shape (..., n, n)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vecs : torch.Tensor
        eigenvector with shape (..., n)
    eigen_vals : torch.Tensor
        eigenvalue with shape (...)
    """
    # Consts
    n = M.shape[-1]
    assert M.shape[-2] == n
    assert M.ndim >= 2

    # # Power method - min eigen
    # I = torch.eye(n, dtype=torch.complex64, device=M.device)
    # eigen_vecs = torch.ones((*M.shape[:-1], 1), device=M.device, dtype=M.dtype)
    # npower = num_iter * 2
    # for i in tqdm(range(npower), 'Power Iterations', disable=not verbose):
    #     if i == npower // 2:
    #         M = eigen_vals * I - M
    #         eigen_vals_mx = eigen_vals.clone()
    #         eigen_vecs = eigen_vecs * 0 + 1
    #     eigen_vecs = M @ eigen_vecs
    #     eigen_vals = torch.linalg.norm(eigen_vecs, axis=-2, keepdims=True)
    #     eigen_vecs = eigen_vecs / eigen_vals
    # eigen_vecs = rearrange(eigen_vecs, '... nc 1 -> nc ...')
    # eigen_vals = (eigen_vals_mx - eigen_vals).squeeze()

    eigen_vecs = torch.ones((*M.shape[:-1], 1), device=M.device, dtype=M.dtype)
    for i in tqdm(range(num_iter), 'Power Iterations', disable=not verbose):
        eigen_vecs = M @ eigen_vecs
        eigen_vals = torch.linalg.norm(eigen_vecs, axis=-2, keepdims=True)
        eigen_vecs = eigen_vecs / eigen_vals
    eigen_vecs = rearrange(eigen_vecs, '... n 1 -> n ...')
    
    return eigen_vecs, eigen_vals.squeeze()