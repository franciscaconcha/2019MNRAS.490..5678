"""
This script runs the viscous ring evolution test, following the
analytic solution from Pringle (1981, ARAA, 19, 137).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader.test import ring
sys.path.pop()
sys.path.pop()

# Rebuild library
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib PROB=ring')

# Set parameter file name
paramFile = os.path.join(_path, 'vader', 'test', 'ring.param')

# Run the simulation, printing run time
startime = time.clock()
x, tOut, colOut, colExact, err, presOut, l1err \
    , mBndOut, mDiskOut, eBndOut, eDiskOut, \
    nStep, nIter, nFail \
    = ring(paramFile, nSave=65)
endtime = time.clock()
print("Computation done in {:f} seconds.".format(endtime-startime))

# Plot exact vs. analytic solution
plt.figure(figsize=(8,8))
skip=len(x)/64
ax=plt.subplot(2,1,1)
outSteps=[2,4,16,64]
colors=['k', 'r', 'g', 'b']
plots1=[]
plots2=[]
labels=[]
for i, s in enumerate(outSteps):
    p1,=plt.plot(x, colExact[s,:], colors[i], lw=2)
    p2,=plt.plot(x[::skip], colOut[s,::skip], colors[i]+'o', lw=2)
    plots1.append(p1)
    plots2.append(p2)
    labels.append(r'$t/t_0 = {:5.3f}$'.format(tOut[s]))
plt.yscale('log')
plt.xlim([0,2.])
plt.ylim([1e-6,1e7])
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel(r'$\Sigma/\Sigma_0$')
leg1=plt.legend(plots1, labels, title='Analytic', loc=1)
leg2=plt.legend(plots2, labels, title='Numerical', loc=2, numpoints=1)
ax.add_artist(leg1)

# Plot errors
plt.subplot(2,1,2)
for i, s in enumerate(outSteps):
    plt.plot(x, np.abs(err[s,:]), colors[i], lw=2)
plt.xlim([0,2.])
plt.yscale('log')
plt.ylim([1e-6, 3e-2])
plt.xlabel(r'$r/R_0$')
plt.ylabel('Error')

# Adjust plots
plt.subplots_adjust(hspace=0)

# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Save
print("Test complete. Writing output/ring1.pdf.")
fname = os.path.join(dirname, 'ring1')
plt.savefig(fname+'.pdf')

