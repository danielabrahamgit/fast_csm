import numpy as np
import sigpy as sp
import sigpy.mri as mri
import torch
import os
import time

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.utils.func import np_to_torch, torch_to_np
from fast_csm.train_kernels import train_kernels, gen_source_vectors_rot
from fast_csm.extract_mps import csm_from_kernels
from sigpy.mri.app import EspiritCalib
from mr_sim.trj_lib import trj_lib
from mr_recon.utils.func import normalize, calc_coil_subspace, fourier_resize

# Set seed
np.random.seed(100)
torch.manual_seed(100)

# GPU/CPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device_idx = 5
device = sp.Device(device_idx)
mvd = lambda x : sp.to_device(x, device)
mvc = lambda x : sp.to_device(x, sp.cpu_device)
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')
xp = device.xp
print(device)

# Simulation params
ndim = 2
im_size = (220,) * ndim
n_coil = 12
calib_size = (32,) * ndim
coil_compress = False
trj_type = 'spi'
sigma = 5e-3 * 0
rng = np.random.default_rng(100)

# Recon params
max_iter = 100
lamda_l2 = 1e-2

# Gen Phantom
phantom = sp.shepp_logan(im_size, dtype=np.complex64)

# Gen sense maps
mps = mri.birdcage_maps((n_coil, *im_size), r=1.25, dtype=np.complex64)
# mps = np.load('../mr_data/dwi_spiral_phantom/data/mps.npy')[..., 19]
# mps = np.roll(mps, shift=9, axis=-1)
# plt.imshow(np.mean(np.abs(mps[:, ..., 19]), axis=0) * phantom.real, )
# plt.show()
# quit()

# Synthesize calibration
trj_cal = trj_lib.acs(calib_size)
ksp_cal = mri.linop.Sense(mps, trj_cal) * phantom

if coil_compress:
    sub, ksp_cal, mps = calc_coil_subspace(ksp_cal, n_coil, ksp_cal, mps)

ksp_cal_noisy = ksp_cal + rng.normal(0, sigma, ksp_cal.shape) + 1j * rng.normal(0, sigma, ksp_cal.shape)

# Show shapes
print(f'ksp_cal shape = {ksp_cal.shape}')
print(f'mps shape = {mps.shape}')

# Implicit estimation
M = 1
im_size_lowres = [im_size[i] // M for i in range(ndim)]
img_cal = sp.ifft(ksp_cal_noisy, axes=range(-ndim, 0))
source_vecs = gen_source_vectors_rot(num_kerns=5000,
                                     num_inputs=5,
                                     ofs=0.15,
                                     line_width=2.0)
source_vecs = np_to_torch(source_vecs).to(torch_dev)
img_cal = np_to_torch(img_cal).to(torch_dev)
start = time.perf_counter()
grappa_kernels = train_kernels(img_cal, source_vecs, 
                               lamda_tikonov=1e0,
                               solver='lstsq_torch',
                               fast_method=False,)
mps_impl, vals_impl = csm_from_kernels(grappa_kernels, source_vecs, im_size_lowres, verbose=True)
end = time.perf_counter()
mps_impl = mps_impl.cpu()
vals_impl = vals_impl.cpu()
impl_time = end - start

# Estimate maps via ESPRIT
start = time.perf_counter()
mps_esp, vals_esp = EspiritCalib(sp.resize(ksp_cal, (n_coil, *im_size_lowres)), 
                                 calib_width=calib_size[0], 
                                 crop=0.0,
                                 device=sp.Device(device_idx),
                                 output_eigenvalue=True).run()
end = time.perf_counter()
mps_esp = fourier_resize(mps_esp, im_size)
vals_esp = fourier_resize(vals_esp, im_size)
mps_esp = sp.to_device(mps_esp) * M
vals_esp = sp.to_device(vals_esp)[0] * M
esp_time = end - start

# Times
print(f'ESPIRIT time = {esp_time:.3f}s')
print(f'Implicit time = {impl_time:.3f}s')

# Plot
msk = np.abs(phantom) > 0
# mps = mps * msk
# mps_impl = mps_impl * msk
# mps_esp = mps_esp * msk
# vals_impl = vals_impl * msk
# vals_esp = vals_esp * msk
for i in range(3):
    if ndim == 3:
        slc = im_size[-1]//2
        tup = (i,) + (slice(None),) * (ndim - 1) + (slc,)
        tup_vals = (slice(None),) * (ndim - 1) + (slc,)
    else:
        tup = (i,) + (slice(None),) * ndim
        tup_vals = (slice(None),) * ndim
    vmin = np.abs(mps[tup] * msk).min()
    vmax = np.abs(mps[tup] * msk).max()
    plt.figure(figsize=(14,7))
    plt.subplot(231)
    plt.title('Ground Truth')
    plt.imshow(np.abs(mps[tup]), cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplot(232)
    plt.title(f'Implicit Estimate ({impl_time:.3f}s)')
    plt.imshow(np.abs(mps_impl[tup]), cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplot(233)
    plt.title(f'ESPIRIT Estimate ({esp_time:.3f}s)')
    plt.imshow(np.abs(mps_esp[tup]), cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplot(235)
    plt.title(f'Implicit lamda')
    plt.imshow(np.abs(vals_impl[tup_vals]), cmap='gray')
    plt.axis('off')
    plt.subplot(236)
    plt.title(f'ESPIRIT lamda')
    plt.imshow(np.abs(vals_esp[tup_vals]), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
plt.show()
quit()
