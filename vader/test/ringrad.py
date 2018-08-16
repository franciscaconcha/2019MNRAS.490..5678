"""
This script runs the viscous ring evolution test, following the
analytic solution from Pringle (1981, ARAA, 19, 137). The equation of
state is set to a complex one including both gas and radiation
pressure.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader.test import ringrad
import vader
sys.path.pop()
sys.path.pop()

# Rebuild library
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib PROB=ringrad')

# Set parameter file name
paramFile = os.path.join(_path, 'vader', 'test', 'ringrad.param')

# Run the simulation without radiation pressure
startime = time.clock()
print("Starting computation without radiation pressure...")
xNRP, tOutNRP, colOutNRP, colExactNRP, errNRP, presOutNRP, \
    pGasOutNRP, pRadOutNRP, eIntOutNRP, eGravOutNRP, eOrbOutNRP, \
    eOutNRP, tempOutNRP, eDiskOutNRP, eBndOutNRP, \
    nStepNRP, nIterNRP, nFailNRP \
    = ringrad(paramFile, nSave=65, gamma=5./3., delta=0.0,
              f_z0 = 0.0)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))
eConsNRP = eDiskOutNRP-eBndOutNRP[:,0]+eBndOutNRP[:,1]

# Run the simulation with radiation pressure
startime = time.clock()
print("Starting computation with radiation pressure...")
x, tOut, colOut, colExact, err, presOut, pGasOut, pRadOut, \
    eIntOut, eGravOut, eOrbOut, eOut, tempOut, eDiskOut, eBndOut, \
    nStep, nIter, nFail \
    = ringrad(paramFile, nSave=65)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))
eCons = eDiskOut-eBndOut[:,0]+eBndOut[:,1]

# Grab some data from parameter file
paramDict = vader.readParam(paramFile)
fz0 = paramDict['f_z0']

# Plot pressure in no radiation run
plt.figure(figsize=(8,8))
ax=plt.subplot(2,1,1)
outSteps=[1,16,64]
colors=['k', 'r', 'g', 'b', 'c']
plots1=[]
labels=[]
for i, s in enumerate(outSteps):
    p1,=plt.plot(x, presOutNRP[s,:]/fz0, colors[i], lw=2)
    plots1.append(p1)
    labels.append(r'$t/t_0 = {:5.3f}$'.format(tOut[s]))
plt.yscale('log')
plt.ylim([1e3,1e9])
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel(r'$P/f z_0$ [dyn cm$^{-2}$]')
plt.legend(plots1, labels, title='Gas only ($\gamma=5/3$)')

# Plot pressure in radiation run
ax=plt.subplot(2,1,2)
plots1=[]
labels=[]
for i, s in enumerate(outSteps):
    p1,=plt.plot(x, presOut[s,:]/fz0, colors[i], lw=2)
    p2,=plt.plot(x, pGasOut[s,:]/fz0, colors[i]+'--', lw=2)
    p3,=plt.plot(x, pRadOut[s,:]/fz0, colors[i]+':', lw=2)
    if (i == 0):
        plots1=[p1, p2, p3]
        labels=['Total', 'Gas', 'Radiation']
plt.yscale('log')
plt.ylim([1e3,9e8])
plt.xlabel(r'$r/R_0$')
plt.ylabel(r'$P/f z_0$ [dyn cm$^{-2}]$')
plt.legend(plots1, labels, title='Gas plus radiation')

# Adjust plots
plt.subplots_adjust(hspace=0)

# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Save
print("Computation done. Saving output/ringrad1.pdf.")
fname = os.path.join(dirname, 'ringrad1')
plt.savefig(fname+'.pdf')


# Plot temperatures
plt.figure(figsize=(8,8))
ax=plt.subplot(2,1,1)
plots1=[]
labels=[]
for i, s in enumerate(outSteps):
    p1,=plt.plot(x, tempOutNRP[s,:], colors[i], lw=2)
    plots1.append(p1)
    labels.append(r'$t/t_0 = {:5.3f}$'.format(tOut[s]))
plt.yscale('log')
plt.ylim([1e2,1e18])
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel(r'$T_{\rm eff}$ [K]')
plt.legend(plots1, labels, title='Gas only ($\gamma=5/3$)')
ax=plt.subplot(2,1,2)
for i, s in enumerate(outSteps):
    p1,=plt.plot(x, tempOut[s,:], colors[i], lw=2)
plt.yscale('log')
plt.ylim([1e2,9e5])
plt.xlabel(r'$r/R_0$')
plt.ylabel(r'$T_{\rm eff}$ [K]')
plt.legend([], [], title='Gas plus radiation')
plt.subplots_adjust(hspace=0)
print("Writing output/ringrad2.pdf.")
fname = os.path.join(dirname, 'ringrad2')
plt.savefig(fname+'.pdf')

# Print conservation error information
print("Conservation errors:")
print("  constant gamma: max error = {:e}, mean error = {:e}".
      format(np.amax(abs((eConsNRP-eConsNRP[0])/eDiskOutNRP[0])),
             np.mean(abs((eConsNRP-eConsNRP[0])/eDiskOutNRP[0]))))
print("  radiation + gas: max error = {:e}, mean error = {:e}".
      format(np.amax(abs((eCons-eCons[0])/eDiskOut[0])),
             np.mean(abs((eCons-eCons[0])/eDiskOut[0]))))


