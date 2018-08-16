"""
This script runs the Krumholz & Burkert (2010) GI disk test
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader.test import gidisk
sys.path.pop()
sys.path.pop()

# Rebuild library
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib PROB=gidisk')

# Set parameter file name
paramFile = os.path.join(_path, 'vader', 'test', 'gidisk.param')

# First simulation, starting in steady state
startime = time.clock()
print("Starting steady state computation...")
xss, tOutss, colOutss, presOutss, Qss, colSteady, presSteady, \
    nStepss, nIterss, nFailss \
    = gidisk(paramFile, verbosity=1, nSave=17)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))
sigmaOutss = np.sqrt(presOutss/colOutss)
sigmaSteady = np.sqrt(presSteady/colSteady)

# Second simulation, starting at Q = 0.5
startime = time.clock()
print("Starting computation with Q = 0.5...")
xqlo, tOutqlo, colOutqlo, presOutqlo, Qqlo, colSteady, presSteady, \
    nStepqlo, nIterqlo, nFailqlo\
    = gidisk(paramFile, verbosity=1, nSave=1001, n_orbit=1.0,
             init_vdisp=0.5, obc_vdisp=0.5)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))
sigmaOutqlo = np.sqrt(presOutqlo/colOutqlo)

# Third simulation, starting at Q = 1 but with s = 2x steady state value
startime = time.clock()
print("Starting computation with Q = 1, s = 2 x equilibrium...")
xshi, tOutshi, colOutshi, presOutshi, Qshi, colSteady, presSteady, \
    nStephi, nIterhi, nFailhi \
    = gidisk(paramFile, verbosity=1, nSave=41, n_orbit=20.0,
             init_vdisp=2.0, init_col=2.0, obc_vdisp=2.0)
endtime = time.clock()
print("Computation done in {:f} seconds.\n\n".format(endtime-startime))
sigmaOutshi = np.sqrt(presOutshi/colOutshi)

# Plot result of steady state test
# Column density
plt.figure(figsize=(8,8))
ax1=plt.subplot(3,1,1)
outsteps=[0,-1]
plots=[]
labels=[]
p,=plt.plot(xss, colSteady, 'k--', lw=4)
plots.append(p)
labels.append('Steady state')
for s in outsteps:
    if s < len(tOutss):
        p,=plt.plot(xss, colOutss[s,:], lw=2)
        plots.append(p)
        labels.append(r'Simulation, $T = {:4.2f}$'.format(tOutss[s]))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\Sigma/\Sigma(R)$')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.legend(plots, labels)

# Velocity dispersion
ax2=plt.subplot(3,1,2)
plt.plot(xss, sigmaSteady, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutss):
        p,=plt.plot(xss, sigmaOutss[s,:], lw=2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\sigma/v_\phi$')
plt.ylim([1e-2,2e-1])
plt.setp(ax2.get_xticklabels(), visible=False)

# Toomre Q
ax3=plt.subplot(3,1,3)
plt.plot(xss, 0*xss+1, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutss):
        p,=plt.plot(xss, Qss[s,:], lw=2)
plt.xscale('log')
plt.ylim([0,2.25])
plt.xlabel(r'$r/R$')
plt.ylabel(r'$Q$')

plt.subplots_adjust(hspace=0.0)


# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Write output
print("Test complete. Writing ooutput/gidisk1.pdf.")
fname = os.path.join(dirname, 'gidisk1')
plt.savefig(fname+'.pdf')


# Plot result of Q = 0.5 test
# Column density
plt.figure(figsize=(8,8))
ax1=plt.subplot(3,1,1)
outsteps=[0,1,100,1000]
plots=[]
labels=[]
p,=plt.plot(xqlo, colSteady, 'k--', lw=4)
plots.append(p)
labels.append('Steady state')
for s in outsteps:
    if s < len(tOutqlo):
        p,=plt.plot(xqlo, colOutqlo[s,:], lw=2)
        plots.append(p)
        labels.append(r'Simulation, $T = {:5.3f}$'.format(tOutqlo[s]))
plt.xscale('log')
plt.yscale('log')
plt.ylim([1,10.**3.5])
plt.ylabel(r'$\Sigma/\Sigma(R)$')
plt.setp(ax1.get_xticklabels(), visible=False)
leg1=plt.legend(plots[:len(plots)/2], labels[:len(labels)/2], loc=2)
leg2=plt.legend(plots[len(plots)/2:], labels[len(labels)/2:], loc=1)
ax1.add_artist(leg1)

# Velocity dispersion
ax2=plt.subplot(3,1,2)
plt.plot(xqlo, sigmaSteady, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutqlo):
        p,=plt.plot(xqlo, sigmaOutqlo[s,:], lw=2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\sigma/v_\phi$')
plt.ylim([1e-2,2e-1])
plt.setp(ax2.get_xticklabels(), visible=False)

# Toomre Q
ax3=plt.subplot(3,1,3)
plt.plot(xqlo, 0*xqlo+1, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutqlo):
        p,=plt.plot(xqlo, Qqlo[s,:], lw=2)
plt.xscale('log')
plt.ylim([0,2.25])
plt.xlabel(r'$r/R$')
plt.ylabel(r'$Q$')

plt.subplots_adjust(hspace=0.0)
print("Writing output/gidisk2.pdf.")
fname = os.path.join(dirname, 'gidisk2')
plt.savefig(fname+'.pdf')


# Plot result of s = 2 s_steady test
# Column density
plt.figure(figsize=(8,8))
ax1=plt.subplot(3,1,1)
outsteps=[0,1,20,40]
plots=[]
labels=[]
p,=plt.plot(xshi, colSteady, 'k--', lw=4)
plots.append(p)
labels.append('Steady state')
for s in outsteps:
    if s < len(tOutshi):
        p,=plt.plot(xshi, colOutshi[s,:], lw=2)
        plots.append(p)
        labels.append(r'Simulation, $T = {:3.1f}$'.format(tOutshi[s]))
plt.xscale('log')
plt.yscale('log')
plt.ylim([1,1e4])
plt.ylabel(r'$\Sigma/\Sigma(R)$')
plt.setp(ax1.get_xticklabels(), visible=False)
leg1=plt.legend(plots[:len(plots)/2], labels[:len(labels)/2], loc=2)
leg2=plt.legend(plots[len(plots)/2:], labels[len(labels)/2:], loc=1)
ax1.add_artist(leg1)

# Velocity dispersion
ax2=plt.subplot(3,1,2)
plt.plot(xshi, sigmaSteady, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutshi):
        p,=plt.plot(xshi, sigmaOutshi[s,:], lw=2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\sigma/v_\phi$')
plt.ylim([1e-2,2e-1])
plt.setp(ax2.get_xticklabels(), visible=False)

# Toomre Q
ax3=plt.subplot(3,1,3)
plt.plot(xshi, 0*xshi+1, 'k--', lw=4)
for s in outsteps:
    if s < len(tOutshi):
        p,=plt.plot(xshi, Qshi[s,:], lw=2)
plt.xscale('log')
plt.ylim([0,2.25])
plt.xlabel(r'$r/R$')
plt.ylabel(r'$Q$')

plt.subplots_adjust(hspace=0.0)
print("Writing output/gidisk3.pdf.")
fname = os.path.join(dirname, 'gidisk3')
plt.savefig(fname+'.pdf')

