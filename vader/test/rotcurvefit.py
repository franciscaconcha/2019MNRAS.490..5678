"""
This script checks the rotation curve interpolation routines
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from vader import grid
sys.path.pop()
sys.path.pop()

# Make sure library is compiled
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make lib')

# Generate a parm dict
pD = { 'nr' : 512,
       'grid_type' : 'logarithmic' }

# Data set 1: Paczynski-Witta potential from 2-10 r_g, sampled at intervals of 0.1
r = np.arange(1.9, 10.11, 0.1)
vphi = r**0.5/(r-1.0)
pD['rmin'] = 2.0
pD['rmax'] = 10.0
rotCurveTab=np.zeros((2,len(r)))
rotCurveTab[0,:]=r
rotCurveTab[1,:]=vphi
grdpw = grid(pD, rotCurveTab=rotCurveTab)
psiExact = -1.0/(grdpw.r-1.0)
vphiExact = grdpw.r**0.5/(grdpw.r-1.0)
betaExact = 0.5 * (1.0+grdpw.r) / (1.0-grdpw.r)

# Data set 2: tabulated MW rotation curve from Bhattacharjee et al. (2014, ApJ, 785, 63)
mwdata=np.array([
    [8.5, 220,   0.19, 227.21,   9.95],
    [8.5, 220,   0.35, 265.17,   8.81],
    [8.5, 220,   0.63, 252.45,  13.31],
    [8.5, 220,   1.63, 212.09,   6.1],
    [8.5, 220,   2.56, 211.79,   1.52],
    [8.5, 220,   3.52, 214.08,   2.62],
    [8.5, 220,   4.52, 233.16,   3.43],
    [8.5, 220,   5.58, 236.38,   1.79],
    [8.5, 220,   6.55, 250.82,   2.23],
    [8.5, 220,   7.57, 247.49,   0.85],
    [8.5, 220,   8.42, 247.03,   0.62],
    [8.5, 220,   9.50, 210.12,   8.28],
    [8.5, 220,  10.46, 209.79,  12.54],
    [8.5, 220,  11.42, 225.16,  12.25],
    [8.5, 220,  12.51, 245.87,  21.24],
    [8.5, 220,  13.36, 247.75,  19.96],
    [8.5, 220,  14.44, 216.31,  14.91],
    [8.5, 220,  16.19, 291.96,  25.91],
    [8.5, 220,  19.14, 183.02,  54.32],
    [8.5, 220,  26.30, 193.97,   6.5],
    [8.5, 220,  28.27, 232.77,  11.34],
    [8.5, 220,  29.51, 200.52,   7.49],
    [8.5, 220,  32.02, 168.78,   4.34],
    [8.5, 220,  34.01, 170.12,   7.23],
    [8.5, 220,  35.99, 199.64,  13.59],
    [8.5, 220,  38.08, 176.82,   7.05],
    [8.5, 220,  40.45, 199.10,  14.07],
    [8.5, 220,  42.43, 188.84,   7.21],
    [8.5, 220,  44.49, 212.10,  17.78],
    [8.5, 220,  45.99, 182.11,  11.14],
    [8.5, 220,  48.05, 206.71,  23.12],
    [8.5, 220,  49.53, 181.39,  18.16],
    [8.5, 220,  51.93, 175.63,  19.33],
    [8.5, 220,  54.38, 153.73,  22.55],
    [8.5, 220,  56.94, 191.44,  24.88],
    [8.5, 220,  57.99, 208.52,  34.05],
    [8.5, 220,  60.93, 169.75,  25.57],
    [8.5, 220,  64.70, 204.62,  36.72],
    [8.5, 220,  69.27, 204.21,  39.94],
    [8.5, 220,  73.03, 195.35,  42.13],
    [8.5, 220,  77.01, 232.18,  77.77],
    [8.5, 220,  81.23, 176.35,  62.77],
    [8.5, 220,  85.04, 124.68,  37.32],
    [8.5, 220,  89.44, 137.89,  29.83],
    [8.5, 220,  92.56, 158.21,  68.98],
    [8.5, 220,  97.58, 188.51,  74.84],
    [8.5, 220, 100.87, 105.18,  39.57],
    [8.5, 220, 106.92, 150.54,  58.93],
    [8.5, 220, 120.01, 139.31,  24.50],
    [8.5, 220, 189.52, 128.24,  40.03]
  ])
MWrotCurve = np.zeros((2,50))
MWrotCurve[0,:] = mwdata[:,2]
MWrotCurve[1,:] = mwdata[:,3]
pD['rmin'] = 0.2
pD['rmax'] = 180.0
pD['bspline_degree'] = 2
grdmw2 = grid(pD, rotCurveTab=MWrotCurve)
pD['bspline_degree'] = 3
grdmw3 = grid(pD, rotCurveTab=MWrotCurve)
pD['bspline_degree'] = 4
grdmw4 = grid(pD, rotCurveTab=MWrotCurve)
pD['bspline_degree'] = 3
pD['bspline_breakpoints'] = 8
grdmw3a = grid(pD, rotCurveTab=MWrotCurve)

# Plot
colors=['b', 'g', 'r']

plt.figure(figsize=(5,8))
ax=plt.subplot(2,1,1)

step=16
p1,=plt.plot(grdpw.r, vphiExact, colors[0], lw=2)
p2,=plt.plot(grdpw.r, psiExact, colors[1], lw=2)
p3,=plt.plot(grdpw.r, betaExact, colors[2], lw=2)
p1a,=plt.plot(grdpw.r[::step], grdpw.vphi[::step], colors[0]+'o')
p2a,=plt.plot(grdpw.r[::step], grdpw.psi[::step]-grdpw.psi[-1]+psiExact[-1],
         colors[1]+'o')
p3a,=plt.plot(grdpw.r[::step], grdpw.beta[::step], colors[2]+'o')
plots1=[p1,p2,p3]
labels=[r'$v_\phi$', r'$\psi$', r'$\beta$']
plots2=[p1a,p2a,p3a]
leg1=plt.legend(plots1, labels, title='Analytic', loc=9)
leg2=plt.legend(plots2, labels, title='Fit', loc=1, numpoints=1)
ax.add_artist(leg1)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel(r'$v_\phi$, $\psi$, $\beta$')
plt.ylim([-2,3])

ax=plt.subplot(2,1,2)
p1,=plt.plot([0,1], [0,1], 'k', lw=2)
p2,=plt.plot([0,1], [0,1], 'k--', lw=2)
plt.plot(grdpw.r, (grdpw.vphi-vphiExact)/vphiExact, colors[0], lw=2)
plt.plot(grdpw.r, -(grdpw.vphi-vphiExact)/vphiExact, colors[0]+'--', lw=2)
plt.plot(grdpw.r, (grdpw.psi-grdpw.psi[-1]-psiExact+psiExact[-1])/psiExact,
         colors[1], lw=2)
plt.plot(grdpw.r, -(grdpw.psi-grdpw.psi[-1]-psiExact+psiExact[-1])/psiExact,
         colors[1]+'--', lw=2)
plt.plot(grdpw.r, (grdpw.beta-betaExact)/betaExact, colors[2], lw=2)
plt.plot(grdpw.r, -(grdpw.beta-betaExact)/betaExact, colors[2]+'--', lw=2)
plt.xlim([2,10])
plt.ylim([1e-12,8e-2])
plots=[p1,p2]
labels=[r'Error > 0', r'Error < 0']
plt.legend(plots, labels, loc=1)
plt.yscale('log')
plt.ylabel('|Error|')
plt.xlabel(r'$r/r_g$')

plt.subplots_adjust(hspace=0.0, left=0.18)

# Create directoy into which output can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Save output
print("Writing output/rotcurve1.pdf")
fname = os.path.join(dirname, 'rotcurve1')
plt.savefig(fname+'.pdf')


plt.figure(figsize=(5,8))
ax=plt.subplot(3,1,1)
p1,=plt.plot(MWrotCurve[0,:], MWrotCurve[1,:], 'ks')
plt.errorbar(MWrotCurve[0,:], MWrotCurve[1,:], yerr=mwdata[:,-1],
                 ecolor='k', elinewidth=1, fmt=None)
p2,=plt.plot(grdmw2.r, grdmw2.vphi, colors[0]+'--', lw=2)
p3,=plt.plot(grdmw3.r, grdmw3.vphi, colors[0], lw=2)
p4,=plt.plot(grdmw4.r, grdmw4.vphi, colors[0]+':', lw=2)
p3a,=plt.plot(grdmw3a.r, grdmw3a.vphi, colors[0], lw=1)
plt.xscale('log')
plt.ylim([0,700])
plots=[p1,p2, p4]
labels=['Data', r'$D=2$', r'$D=4$']
leg1=plt.legend(plots, labels, numpoints=1, loc=2)
plots=[p3,p3a]
labels=[r'$D=3$, $B=15$', r'$D=3$, $B=8$']
plt.legend(plots, labels, numpoints=1)
ax.add_artist(leg1)
plt.ylabel(r'$v_\phi$ [km s$^{-1}$]')
plt.setp(ax.get_xticklabels(), visible=False)
plt.xlim([0.1,5e2])

ax=plt.subplot(3,1,2)
plt.plot(grdmw2.r, (grdmw2.psi-grdmw2.psi[-1])/200.**2,
         colors[0]+'--', lw=2)
plt.plot(grdmw3.r, (grdmw3.psi-grdmw3.psi[-1])/200.**2,
         colors[0], lw=2)
plt.plot(grdmw4.r, (grdmw4.psi-grdmw4.psi[-1])/200.**2,
         colors[0]+':', lw=2)
plt.plot(grdmw3a.r, (grdmw3a.psi-grdmw3a.psi[-1])/200.**2,
         colors[0], lw=1)
plt.xscale('log')
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel(r'$\psi/(200\,\mathrm{km}\,\mathrm{s}^{-1})^2$')
plt.xlim([0.1,5e2])

ax=plt.subplot(3,1,3)
plt.plot(grdmw2.r, grdmw2.beta, colors[0]+'--', lw=2)
plt.plot(grdmw3.r, grdmw3.beta, colors[0], lw=2)
plt.plot(grdmw4.r, grdmw4.beta, colors[0]+':', lw=2)
plt.plot(grdmw3a.r, grdmw3a.beta, colors[0], lw=1)
plt.xscale('log')
plt.xlabel('r [kpc]')
plt.ylabel(r'$\beta$')
plt.ylim([-3,1.7])
plt.xlim([0.1,5e2])

plt.subplots_adjust(hspace=0.0, left=0.15)

print("Writing output/rotcurve2.pdf")
fname = os.path.join(dirname, 'rotcurve2')
plt.savefig(fname+'.pdf')

