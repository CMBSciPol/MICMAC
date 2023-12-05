# Cornerplot script

from __future__ import print_function
import healpy as hp
import pysm3
from scipy import interpolate
from scipy.stats import norm
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator
import time
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython
import numpy as np
#from corner import corner

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "light"
plt.rc('text', usetex=True)



def gaussian(all_samples,x0,sigma):
  return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((all_samples - x0)/sigma, 2.)/2.)

def multivariate_gaussian(all_samples, mean, cov):
  n = mean.shape[0]
  cov_det = np.linalg.det(cov)
  cov_inv = np.linalg.inv(cov)
  N = np.sqrt((2*np.pi)**n * cov_det)
  # This einsum call calculates (all_samples-mu)T.Sigma-1.(all_samples-mu) in a vectorized
  # way across all the input variables.
  fac = np.einsum('...k,kl,...l->...', all_samples-mean, cov_inv, all_samples-mean)
  return np.exp(-fac / 2) / N

  
def gaussian_contour_2d(ax2d, mean, cov, levels):
  xmin, xmax = ax2d.get_xlim()
  ymin, ymax = ax2d.get_ylim()
  x_array = np.linspace(xmin, xmax, 1000)
  y_array = np.linspace(ymin, ymax, 1000)
  xy_array = np.meshgrid(x_array, y_array)
  xy_flat = np.vstack((xy_array[0].flatten(), xy_array[1].flatten()))
  z_array = multivariate_gaussian(np.swapaxes(xy_flat, 0, 1), mean, cov).reshape((x_array.size, y_array.size))

  # the contour plot:
  n = 1000
  z_array = z_array/z_array.sum()
  t = np.linspace(0, z_array.max(), n)
  integral = ((z_array >= t[:, None, None]) * z_array).sum(axis=(1,2))
  
  f = interpolate.interp1d(integral, t)
  t_contours = f(np.array(levels))
  #ax2d.contour(xy_array[1], xy_array[0], z_array, t_contours, linewidths=2.0)
  return xy_array, z_array, t_contours



infile = ['file1', 'file2']

x0 = [ ] # True parameters
lab = [ ] # labels of the parameters
names = [] # name of the parameters


all_samples = np.asarray([np.genfromtxt(infile[i]) for i in np.arange(len(infile))])
levels = [0.95, 0.66]
x_gd = [MCSamples(samples=all_samples[i], names=names, labels=lab) for i in np.arange(all_samples.shape[0])]

g=plots.GetDistPlotter()
g.settings.norm_1d_density=True
g.settings.axes_labelsize=15
g.settings.legend_fontsize=30
g.settings.figure_legend_loc='upper right'
g.triangle_plot(x_gd, filled=True)

# Extract the axes
axes = np.array(g.subplots).reshape((len(x0), len(x0)))

# Get Fisher and covariance
fisher = # Fisher matrix
cov = np.linalg.inv(fisher)
mean = np.zeros(len(x0))

# Loop over the diagonal
for i in range(len(x0)):
    ax = axes[i, i]
    xinf, xsup = ax.get_xlim()
    x_fish = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    ax.clear()
    ax.set_xlim(xinf, xsup)
    ax.hist(all_samples, 20, density=True, histtype='step', fill=False, linewidth=2.0)
    ax.axvline(x0[i], color="crimson", linestyle='--')
    mean[i] = norm.fit(all_samples[:, i], scale=np.sqrt(cov[i, i]))[0]
    ax.plot(x_fish, gaussian(x_fish, mean[i], np.sqrt(cov[i, i])), 'k', label='Fisher')
    handles, labels = ax.get_legend_handles_labels()

    if i == len(x0) - 1:
      ax.set_xlabel(names[-1])

# Loop over the histograms
for yi in range(len(x0)):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(x0[xi], color="crimson", linestyle='--')
        ax.axhline(x0[yi], color="crimson", linestyle='--')
        ax.plot(x0[xi], x0[yi], "s", color='crimson')
        #xy_array, z_array, t_contours = gaussian_contour_2d(ax, np.array((mean[xi], mean[yi])), np.array(((cov[xi, xi], cov[xi, yi]), (cov[yi, xi], cov[yi, yi]))), levels)
        #ax.contour(xy_array[0], xy_array[1], z_array, t_contours, linewidths=2.0, colors=['k', 'k'])

fig = plt.gcf()
fig.legend(handles, labels, loc='upper right', fontsize=30)

plt.show()
