import torch
import sigpy as sp

from typing import Optional
from einops import rearrange, einsum
from mr_recon.utils.func import np_to_torch, torch_to_np, lin_solve, rotation_matrix, sp_fft
from sigpy.fourier import _get_oversamp_shape, _apodize, _scale_coord

def gen_source_vectors_rot(num_kerns: int,
                           num_inputs: int,
                           ndim: int,
                           ofs: Optional[float] = 0.15,
                           line_width: Optional[float] = 2.0) -> torch.Tensor:
    """
    Generates a line of source points of width line_width with 
    some desitance ofs from the target points, and rotates about 
    the target to generate mulitple kernel vectors

    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    num_inputs : int
        number of source point inputs
    ndim : int  
        number of dimensions (2 or 3 usually)
    ofs : float
        offset of closest source point to target point
    line_width : float
        width of line of source points
    
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    """

    # Make line
    assert ndim >= 2
    source_vectors = torch.zeros(num_kerns, num_inputs, ndim)
    source_vectors[:, :, 0] = torch.linspace(-line_width / 2, line_width / 2, num_inputs)
    source_vectors[:, :, 1] = ofs

    # Rotate
    if ndim == 2:
        thetas = torch.arange(num_kerns) * 2 * torch.pi / num_kerns
        axis = torch.tensor([0.0, 0.0, 1.0])[None,]
        R = rotation_matrix(axis=axis,
                            theta=thetas)[..., :2, :2]
        source_vectors = einsum(R, source_vectors, 'N to ti, N ninp ti -> N ninp to')
    elif ndim == 3:
        thetas_x = torch.rand(num_kerns) * 2 * torch.pi
        thetas_y = torch.rand(num_kerns) * 2 * torch.pi
        thetas_z = torch.rand(num_kerns) * 2 * torch.pi
        R_x = rotation_matrix(axis=torch.tensor([1.0, 0.0, 0.0])[None,],
                              theta=thetas_x)
        R_y = rotation_matrix(axis=torch.tensor([0.0, 1.0, 0.0])[None,],
                              theta=thetas_y)
        R_z = rotation_matrix(axis=torch.tensor([0.0, 0.0, 1.0])[None,],
                              theta=thetas_z)
        R = einsum(R_z, R_y, R_x, 'N to1 to2, N to2 to3, N to3 to4 -> N to1 to4')
        source_vectors = einsum(R, source_vectors, 'N to ti, N ninp ti -> N ninp to')
    else:
        raise NotImplementedError
    
    return source_vectors

def rect_trj(cal_shape: tuple, 
             dk_buffer: Optional[int] = 4) -> torch.Tensor:
    """
    Creates a recti-linear trajectory.

    Parameters
    ----------
    cal_shape: tuple
        Shape of calibration, something like (nc, nx, ny, (nz))
    dk_buffer: int
        Edge points to remove from calibration

    Returns
    ---------
    trj_rect: torch.Tensor <float>
        The rect-linear coordinates with shape (ntrj, d)
    """

    d = len(cal_shape)
    rect_size = torch.tensor(cal_shape) - dk_buffer
    trj_rect = torch.zeros((*tuple(rect_size), d))
    
    for i in range(d):
        lin_1d = torch.arange(-rect_size[0]/2, rect_size[0]/2)
        tup = [None,]*d
        tup[i] = slice(None)
        trj_rect[..., i] = lin_1d[tuple(tup)]
    
    trj_rect = trj_rect.reshape((-1, d))

    return trj_rect

def grappa_AHA_AHb(img_cal: torch.Tensor, 
                   source_vectors: torch.Tensor,
                   width: Optional[int] = 6,
                   oversamp: Optional[float] = 1.25) -> (torch.Tensor, torch.Tensor):
    """
    Computes AHA and AHb matrices for grappa kernel estimation using
    NUFFT interpolation on the calibration
    
    Parameters:
    -----------
    img_cal: torch.Tensor
        calibration image with shape (nc, *cal_size)
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    width: int
            kaiser bessel kernel width
    oversamp: float
        kaiser bessel oversampling ratio
    
    Returns:
    ----------
    AHA: np.ndarray <complex>
        grappa calibration gram matrix with shape (nkerns, nc * ninputs, nc * ninputs) 
    AHb: np.ndarray <complex>
        grappa adjoint calibration times target poitns with shape (nkerns, nc * ninputs, nc)
    """

    # Consts
    device = img_cal.device
    beta = torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    n_coil = img_cal.shape[0]
    nkerns, num_inputs, d = source_vectors.shape
    assert device == source_vectors.device

    # Move to cupy
    img_cal_cp = torch_to_np(img_cal)
    source_vectors_cp = torch_to_np(source_vectors)
    target_vectors_cp = torch_to_np(rect_trj(img_cal.shape[1:]).to(device))
    dev = sp.get_device(img_cal_cp)

    # Prepare for KB interp
    with dev:

        # FFT part of NUFFT (copied from sigpy)
        os_shape = _get_oversamp_shape(img_cal_cp.shape, d, oversamp)
        output = img_cal_cp.copy()

        # Apodize
        _apodize(output, d, oversamp, width, beta)

        # Zero-pad
        output /= sp.util.prod(img_cal_cp.shape[-d:]) ** 0.5
        output = sp.util.resize(output, os_shape)

        # FFT
        ksp_cal_pre_KB = torch_to_np(sp_fft(np_to_torch(output), dim=tuple(range(-d, 0)), norm=None))

    # KB interpolation
    with dev:
        
        # Target
        coord_trg = _scale_coord(target_vectors_cp, img_cal_cp.shape, oversamp)
        target = sp.interp.interpolate(
            ksp_cal_pre_KB, coord_trg, kernel="kaiser_bessel", width=width, param=beta
        )
        target /= width**d

        # Source
        source_vectors_cp = source_vectors_cp[:, None, ...] + target_vectors_cp[..., None, :]
        coord_src = _scale_coord(source_vectors_cp, img_cal_cp.shape, oversamp)
        source = sp.interp.interpolate(
            ksp_cal_pre_KB, coord_src, kernel="kaiser_bessel", width=width, param=beta
        )
        source /= width**d
    
    # Compute AHA and AHb
    source = rearrange(np_to_torch(source), 'nc N ncal ninp -> N ncal (nc ninp)')
    target = np_to_torch(target).T # ncal nc
    source_H = torch.moveaxis(source, -2, -1).conj()
    AHA = source_H @ source
    AHb = source_H @ target

    return AHA, AHb

def grappa_AHA_AHb_fast(img_cal: torch.Tensor, 
                        source_vectors: torch.Tensor, 
                        width: Optional[int] = 6, 
                        oversamp: Optional[float] = 1.25) -> (torch.Tensor, torch.Tensor):
        """
        A faster grappa kernel estimation algorithm from:
        Luo, T., Noll, D. C., Fessler, J. A., & Nielsen, J. (2019). 
        A GRAPPA algorithm for arbitrary 2D/3D non-Cartesian sampling trajectories with rapid calibration. 
        In Magnetic Resonance in Medicine. Wiley. https://doi.org/10.1002/mrm.27801
        
        Parameters
        ----------
        img_cal: torch.Tensor
            calibration image with shape (nc, *cal_size)
        source_vectors : ntorch.Tensor
            Coordinates of source relative to target with shape (nkerns, ninputs, d)
        width: int
            kaiser bessel kernel width
        oversamp: float
            kaiser bessel oversampling ratio

        Returns:
        ----------
        AHA: np.ndarray <complex>
            grappa calibration gram matrix with shape (nkerns, nc * ninputs, nc * ninputs) 
        AHb: np.ndarray <complex>
            grappa adjoint calibration times target poitns with shape (nkerns, nc * ninputs, nc)
        """
        
        # Consts
        device = img_cal.device
        beta = torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
        n_coil = img_cal.shape[0]
        nkerns, num_inputs, d = source_vectors.shape
        assert device == source_vectors.device

        # Move to cupy
        img_cal_cp = torch_to_np(img_cal)
        source_vectors_cp = torch_to_np(source_vectors)
        dev = sp.get_device(img_cal_cp)

        # Compute pairwise products of calibrations
        with dev:

            # FFT part of NUFFT (copied from sigpy)
            cross_img_cals = img_cal_cp[None, ...].conj() * img_cal_cp[:, None, ...]
            os_shape = _get_oversamp_shape(cross_img_cals.shape, d, oversamp)
            output = cross_img_cals.copy()

            # Apodize
            _apodize(output, d, oversamp, width, beta)

            # Zero-pad
            output /= sp.util.prod(cross_img_cals.shape[-d:]) ** 0.5
            output = sp.util.resize(output, os_shape)

            # FFT
            cross_ksp_cals = torch_to_np(sp_fft(np_to_torch(output), dim=tuple(range(-d, 0)), norm=None))

        # KB interpolation
        with dev:
            cross_orientations = -source_vectors_cp[:, None, :, :] + source_vectors_cp[:, :, None, :]
            coord_AHA = _scale_coord(cross_orientations, img_cal.shape, oversamp)
            AHA = sp.interp.interpolate(
                cross_ksp_cals, coord_AHA, kernel="kaiser_bessel", width=width, param=beta
            )
            AHA /= width**d

            # NUFFT based interpolation for AHb term
            coord_AHb = _scale_coord(-source_vectors_cp, img_cal.shape, oversamp)
            AHb = sp.interp.interpolate(
                cross_ksp_cals, coord_AHb, kernel="kaiser_bessel", width=width, param=beta
            )
            AHb /= width**d

        # Reshape
        AHb = rearrange(np_to_torch(AHb), 
                        'nco nci N ninp -> N (nci ninp) nco')
        AHA = rearrange(np_to_torch(AHA), 
                        'nco nci N ninpo ninpi -> N (nco ninpo) (nci ninpi)').conj()

        return AHA, AHb

def train_kernels(img_cal: torch.Tensor,
                  source_vectors: torch.Tensor,
                  lamda_tikonov: Optional[float] = 1e-3,
                  solver: Optional[str] = 'lstsq_torch',
                  fast_method: Optional[bool] = False) -> torch.Tensor:
    """
    Trains grappa kernels given calib image and source vectors

    Parameters:
    -----------
    img_cal : torch.Tensor
        calibration image with shape (ncoil, *cal_size)
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    lamda_tikonov : float
        tikonov regularization parameter
    solver : str
        linear system solver from ['lstsq_torch', 'lstsq', 'pinv', 'inv']
    fast_method : bool
        toggles fast AHA AHb computation, only worth it for large calib
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    """

    # Consts
    device = img_cal.device
    ncoil = img_cal.shape[0]
    nkerns, num_inputs, d = source_vectors.shape
    assert device == source_vectors.device

    # Compute AHA and AHb
    if fast_method:
        AHA, AHb = grappa_AHA_AHb_fast(img_cal, source_vectors)
    else:
        AHA, AHb = grappa_AHA_AHb(img_cal, source_vectors)

    # Solve
    grappa_kernels = lin_solve(AHA, AHb, lamda=lamda_tikonov, solver=solver)
    grappa_kernels = rearrange(grappa_kernels, 'B (nci ninp) nco -> B nco nci ninp',
                               nci=ncoil, ninp=num_inputs)
    
    return grappa_kernels
