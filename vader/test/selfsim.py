"""
This script runs the self-similar disk evolution test, following the
analytic solution of Lynden-Bell & Pringle (1974, MNRAS, 168, 603).
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader.test import selfsim
sys.path.pop()
sys.path.pop()

# Rebuild library
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib PROB=selfsim')

# Set parameter file name
paramFile = os.path.join(_path, 'vader', 'test', 'selfsim.param')

# Run the simulation with Crank-Nicolson discretization
print("Running simulation with CN time centering...")
startime = time.clock()
x, tOut, colOut, colExact, err, presOut, l1err, \
    mBndOut, mDiskOut, eBndOut, eDiskOut, nStep, nIter, \
    nFail \
    = selfsim(paramFile, nSave=31)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))

# Run the simulation with backwards-Euler discretization
print("Running simulation with BE time centering...")
startime = time.clock()
xBE, tOutBE, colOutBE, colExactBE, errBE, presOutBE, l1errBE, \
    mBndOutBE, mDiskOutBE, eBndOutBE, eDiskOutBE, nStepBE, \
    nIterBE, nFailBE \
    = selfsim(paramFile, nSave=31, method='BE')
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))

# Run the simulation with Crank-Nicolson discretization and a tigher
# error tolerance
startime = time.clock()
print("Running simulation with CN time centering, tol = 10^-10...")
xTol, tOutTol, colOutTol, colExactTol, errTol, presOutTol, l1errTol, \
    mBndOutTol, mDiskOutTol, eBndOutTol, eDiskOutTol, nStepTol, \
    nIterTol, nFailTol \
    = selfsim(paramFile, nSave=31, err_tol=1e-10)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))

# Plot exact vs. analytic
outSteps=[0,10,20,30]
colors=['k', 'r', 'g', 'b']
plt.figure(figsize=(8,8))
ax1=plt.subplot(2,1,1)
skip = len(x)/32
plots1=[]
plots2=[]
labels=[]
for i, s in enumerate(outSteps):
    if len(tOut) > s:
        p1,=plt.plot(x, colExact[s,:], colors[i], lw=2)
        plots1.append(p1)
        labels.append(r'$t/t_s = {:d}$'.format(int(tOut[s])))
        p2,=plt.plot(x[::skip], colOut[s,::skip], colors[i]+'o')
        plots2.append(p2)
plt.xscale('log')
plt.yscale('log')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel(r'$\Sigma/\Sigma_0$')
plt.xlim([1e-1,20])
plt.ylim([1e-10,20])
leg1=plt.legend(plots1, labels, loc=3, title='Analytic')
leg2=plt.legend(plots2, labels, loc=8, title='Numerical', numpoints=1)
ax1.add_artist(leg1)

# Plot errors
ax2=plt.subplot(2,1,2)
outSteps=[10,20,30]
colors=['r', 'g', 'b']
plots=[]
labels=[]
for i, s in enumerate(outSteps):
    if len(tOut) > s:
        p,=plt.plot(x, np.abs(err[s,:]), colors[i], lw=2)
        plots.append(p)
        labels.append(r'$t/t_s = {:d}$'.format(int(tOut[s])))
if len(tOutBE) > outSteps[-1]:
    p,=plt.plot(xBE, np.abs(errBE[outSteps[-1],:]), 'm', lw=2)
    plots.append(p)
    labels.append(r'$t/t_s = {:d}$, BE method'.format(int(tOut[s])))
if len(tOutTol) > outSteps[-1]:
    p,=plt.plot(xTol, np.abs(errTol[outSteps[-1],:]), 'c', lw=2)
    plots.append(p)
    labels.append(r'$t/t_s = {:d}$'.format(int(tOut[s])) +
                  ', tol = $10^{-10}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r/R_0$')
plt.xlim([1e-1,20])
plt.ylim([1e-7,5e-1])
plt.ylabel('|Error|')
leg1=plt.legend(plots[:-2], labels[:-2], loc=2, numpoints=1)
leg2=plt.legend(plots[-2:], labels[-2:], loc=1, numpoints=1)
ax2.add_artist(leg1)
plt.subplots_adjust(hspace=0)

# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Save
print("Test complete. Writing output/selfsim1.pdf.")
fname = os.path.join(dirname, 'selfsim1')
plt.savefig(fname+'.pdf')
