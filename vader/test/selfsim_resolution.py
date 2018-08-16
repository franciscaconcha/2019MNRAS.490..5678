"""
This script runs the self-similar disk evolution test, following the
analytic solution of Lynden-Bell & Pringle (1974, MNRAS, 168, 603). It
does so at a range of resolutions to study convergence properties.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Rebuild library
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib PROB=selfsim')

# Note the somewhat unusual practice of importing the library not at
# the top of the script, but midway through. We do this so that we
# have a chance to rebuild the c library with the appropriate
# user-defined functions before importing it.

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader.test import selfsim
sys.path.pop()
sys.path.pop()

# Set parameter file name
paramFile = os.path.join(_path, 'vader', 'test', 'selfsim.param')

# List of resolutions to test, and lists of dimensionless output
# column densities, pressures, and times
nOut = 2
res = np.array([64, 128, 256, 512, 1024, 2048])
tOut = []
colOut = []
presOut = []
colExact = []
err = []
x = []
l1err = np.zeros((len(res), nOut))

# Run the simulations and record the results
for i, r in enumerate(res):
    print("Running simulation at resolution "+str(r)+"...")
    x1, tOut1, colOut1, colExact1, err1, presOut1, l1err1, \
        mBndOut, mDiskOut, eBndOut, eDiskOut, nStep, nIter, \
        nFail \
        = selfsim(paramFile, nOut=nOut, nr=r, 
                  end_time=2.0, err_tol=1e-10)
    tOut.append(tOut1)
    colOut.append(colOut1)
    presOut.append(presOut1)
    colExact.append(colExact1)
    err.append(err1)
    x.append(x1)
    l1err[i,:]=l1err1

# Plot errors
plt.figure()
outnum = -1
colors=['r', 'g', 'b', 'm', 'c', 'k']
plots=[]
labels=[]
for i, r in enumerate(res):
    p1,=plt.plot(x[i], np.abs(err[i][outnum,:]), colors[i], lw=2)
    plots.append(p1)
    labels.append(r'$N = {:d}$'.format(r))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r/R_0$')
plt.xlim([1e-1,20])
plt.ylim([1e-8,1e-1])
plt.ylabel('|Error|')
plt.subplots_adjust(hspace=0)
leg1=plt.legend(plots[:3], labels[:3], loc=2)
leg2=plt.legend(plots[3:], labels[3:], loc=9)
plt.gca().add_artist(leg1)

# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Write output
print("Test complete. Writing output/selfsim_resolution1.pdf.")
fname = os.path.join(dirname, 'selfsim_resolution1')
plt.savefig(fname+'.pdf')

# Plot L1 errors versus resolution
plt.figure()
p1,=plt.plot(res, l1err[:,outnum], 'b', lw=2)
p2,=plt.plot(res, l1err[:,outnum], 'bo')
p3,=plt.plot(res, l1err[-1,outnum]*2*(res[-1]/res)**2, 'k--')
plt.xlim([50, 3000])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N$')
plt.ylabel(r'$L^1$ Error')
ticklabels=["{:d}".format(r) for r in res]
plt.xticks(res, ticklabels)
plt.tick_params(axis='x', which='minor', bottom='off', top='off')
plt.legend([(p1,p2), p3], ['Numerical', r'$N^{-2}$'])
print("Writing output/selfsim_resolution2.pdf.")
fname = os.path.join(dirname, 'selfsim_resolution2')
plt.savefig(fname+'.pdf')

# Print best-fitting slope
print("Best-fit slope for convergence rate = {:f}".
      format(np.polyfit(np.log(res[:]), np.log(l1err[:,outnum]), 1)[0]))
