import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from scipy.ndimage import shift
import scipy.fft as fft
from PIL import Image
import time
import numba
import pyfftw
import multiprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable

fringes = np.array(Image.open('fringes.tif'))[:, :, 0].astype(np.float32)
# fringes = np.random.randint(size=(2048, 2048), low=0, high=256)


@numba.jit(nopython=True)
def shift5(arr, numi, numj, fill_value=0):
    """Fast array shifting

    :param np.ndarray arr: Array to shift
    :param int numi: Pixel to shift for row number
    :param int numj: Pixel to shift for column number
    :param fill_value: Filling value
    :return: The shifted array
    :rtype: depends on fill value type np.ndarray[np.float32, ndim=2]

    """
    result = np.empty_like(arr)
    if numi > 0:
        result[:numi, :] = fill_value
        result[numi:, :] = arr[:-numi, :]
    elif numi < 0:
        result[numi:, :] = fill_value
        result[:numi, :] = arr[-numi:, :]
    if numj > 0:
        result[:, :numj] = fill_value
        result[:, numj:] = arr[:, :-numj]
    elif numj < 0:
        result[:, numj:] = fill_value
        result[:, :numj] = arr[:, -numj:]
    else:
        result[:] = arr
    return result


fft_side = pyfftw.empty_aligned(fringes.shape, dtype=np.complex64)
fft_center = pyfftw.empty_aligned(fringes.shape, dtype=np.complex64)
fft_obj_s = pyfftw.builders.fft2(fft_side,
                                 overwrite_input=True,
                                 threads=multiprocessing.cpu_count(),
                                 planner_effort="FFTW_MEASURE")
ifft_obj_s = pyfftw.builders.ifft2(fft_side,
                                   overwrite_input=True,
                                   threads=multiprocessing.cpu_count(),
                                   planner_effort="FFTW_MEASURE")
ifft_obj_c = pyfftw.builders.ifft2(fft_center,
                                   overwrite_input=True,
                                   threads=multiprocessing.cpu_count(),
                                   planner_effort="FFTW_MEASURE")


def get_visibility(frame, fft_side=None, fft_center=None, fft_obj_s=None,
                   ifft_obj_s=None, ifft_obj_c=None):
    """Gets the visibility of a given fringe pattern using Fourier filtering

    :param np.ndarray frame: Fringe pattern.
    :param pyfftw.empty_aligned fft_side: Array to perform fft on for side peak
    :param pyfftw.empty_aligned fft_center: Array to perform fft on for center
                                            peak.
    :param pyfftw.FFTW fft_obj_s: Fft instance for the side peak
    :param pyfftw.FFTW ifft_obj_s: Fft instance for the side peak
    :param pyfftw.FFTW ifft_obj_c: Fft instance for the center peak
    :return: Max of visibility and visibility map
    :rtype: (float, np.ndarray[np.float32, ndim=2])

    """

    if fft_side is not None and fft_center is not None:
        fft_side[:] = frame
        fft_center[:] = frame
    else:
        fft_side = np.copy(frame)
        fft_center = np.copy(frame)
    del frame
    kx = fft.fftshift(np.fft.fftfreq(fft_side.shape[1], 5.5e-6))
    ky = fft.fftshift(np.fft.fftfreq(fft_side.shape[0], 5.5e-6))
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2)
    roi = np.zeros(K.shape, dtype=np.complex64)
    roic = np.zeros(K.shape, dtype=np.complex64)
    roi[Kx > 10e2] = 1
    roic[K <= 10e2] = 1
    if fft_obj_s is not None:
        fft_side = fft.fftshift(fft_obj_s(fft.fftshift(fft_side)))
    else:
        fft_side = fft.fftshift(fft.fft2(fft.fftshift(fft_side)))
    fft_center[:] = fft_side*roic
    fft_side[:] = fft_side*roi
    max = np.where(np.abs(fft_side) == np.max(np.abs(fft_side)))
    fft_side = shift5(fft_side, fft_side.shape[0]//2-max[0][0],
                      fft_side.shape[1]//2-max[1][0])
    if ifft_obj_s is not None:
        vis = fft.fftshift(ifft_obj_s(fft.ifftshift(fft_side)))
    else:
        vis = fft.fftshift(fft.ifft2(fft.ifftshift(fft_side)))
    if ifft_obj_c is not None:
        fft_center = fft.fftshift(ifft_obj_c(fft.ifftshift(fft_center)))
    else:
        fft_center = fft.fftshift(fft.ifft2(fft.ifftshift(fft_center)))
    vis /= fft_center
    # filter bad pixels
    vis[:10, :] = 0
    vis[:, :10] = 0
    vis[-10:, :] = 0
    vis[:, -10:] = 0
    return np.max(np.abs(vis[np.abs(vis) > 0.1])), np.abs(vis)

t0 = time.time()
vis, Vis = get_visibility(fringes, fft_side, fft_center, fft_obj_s, ifft_obj_s,
                          ifft_obj_c)
t1 = time.time()-t0
print(f"Fringes visibility is {'{:.2%}'.format(vis)}  (took {'{:.4}'.format(t1)} s)")
max = np.where(Vis == np.max(Vis))
plt.imshow(Vis, vmin=0, vmax=1)
plt.scatter(max[1][0], max[0][0], color='r')
plt.show()
# fig = plt.figure()
# gs = fig.add_gridspec(2, 2)
# ax = []
# ax.append(fig.add_subplot(gs[0, 0]))
# ax.append(fig.add_subplot(gs[0, 1]))
# # spans two rows:
# ax.append(fig.add_subplot(gs[1, :]))
# xs = list(range(0, 200))
# ys = [0] * len(xs)
# ax[2].set_ylim((0, 100))
# im = ax[0].imshow(fringes, vmin=0, vmax=255, cmap='gray')
# im1 = ax[1].imshow(Vis, vmin=0, vmax=1)
# line, = ax[2].plot(xs, ys)
# cbar = fig.colorbar(im, ax=ax[0])
# cbar1 = fig.colorbar(im1, ax=ax[1])
# cbar.set_label("Intensity", rotation=270)
# cbar1.set_label("Visibility", rotation=270)
#
# ax[0].set_title("Camera")
# ax[1].set_title("Fringe visibility")
# ax[2].set_title("Fringe visibility")
# ax[2].set_xlabel("Samples")
# ax[2].set_ylabel("Visibility in %")
# plt.tight_layout()
#
# def animate(i, ys, fft_side, fft_center, fft_obj_s, ifft_obj_s,
#             ifft_obj_c):
#     # frame = np.random.randint(size=(2048, 2048), low=0, high=256)
#     frame = fringes
#     vis, Vis = get_visibility(frame, fft_side, fft_center, fft_obj_s,
#                               ifft_obj_s, ifft_obj_c)
#     im.set_data(frame)
#     im1.set_data(Vis)
#     # Add y to list
#     ys.append(100*vis)
#
#     # Limit y list to set number of items
#     ys = ys[-len(xs):]
#
#     # Update line with new Y values
#     line.set_ydata(ys)
#
#     return im, im1, line,
#
#
# ani = FuncAnimation(fig,
#                     animate,
#                     fargs=(ys, fft_side, fft_center, fft_obj_s, ifft_obj_s,
#                            ifft_obj_c),
#                     interval=50,
#                     blit=True)
# plt.show()
