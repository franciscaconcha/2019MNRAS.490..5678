"""
This script runs an automated performance test on the specified problem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import timeit
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
import vader
sys.path.pop()
sys.path.pop()

# Physical constants in CGS units
import scipy.constants as physcons
kB = physcons.k*1e7
mH = 1e3*physcons.physical_constants['atomic mass constant'][0]*1.00794
mu = 0.61
sigma = physcons.sigma*1e3
c = physcons.c*1e2
a = 4*sigma/c
G = physcons.G*1e3

# Read arguments to figure out which problem to run, and what Anderson
# acceleration parameter to use
probname = sys.argv[1]
aa_m = sys.argv[2]
logdt = sys.argv[3]
dt=10.**float(logdt)
max_iter = 100
max_iter1 = 40
max_step = 1000

# Rebuild library in testing mode, the load it
print("Rebuilding c library...")
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'
os.system('cd '+_path+'; make clean')
buildstr = 'make lib MODE=TEST PROB='+probname+' AA_M='+aa_m
os.system('cd '+_path+'; '+buildstr)
vader.interface.loadlib()

# Read parameter file
paramFile = os.path.join(_path, 'vader', 'test', probname+'.param')
paramDict = vader.readParam(paramFile)

# Initialize the problem
if probname == 'selfsim':

    R0 = paramDict['R0']
    ts = R0**2/(3*paramDict['nu0'])
    grd = vader.grid(paramDict)
    x = grd.r/R0
    col1 = paramDict['Mdot0']/(3*np.pi*paramDict['nu0'])
    col = col1 * np.exp(-x)/x
    pres = col*paramDict['init_temp']*kB/(mu*mH)
    pres1 = col1*paramDict['init_temp']*kB/(mu*mH)
    eInt = None
    tstart = ts
    tend = paramDict['end_time']*ts
    dt = dt*ts
    paramArr = np.array([paramDict['nu0'], paramDict['R0'],
                         paramDict['Mdot0'], 
                         paramDict['init_temp']*kB/(mu*mH)], 
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)

elif probname == 'ring' :

    r0 = paramDict['ring_loc']
    t0 = r0**2/(12.0*paramDict['kinematic_visc'])
    col0 = paramDict['ring_mass'] / (np.pi * r0**2)
    pres0 = col0 * paramDict['init_temp']*kB/(mu*mH)
    grd = vader.grid(paramDict)
    idx = np.argmax(grd.r_h > r0) - 1
    init_col = paramDict['ring_mass'] / grd.area[idx]
    col = np.ones(grd.nr)*init_col/paramDict['col_ratio']
    col[idx] = init_col
    pres = col*paramDict['init_temp']*kB/(mu*mH)
    paramArr = np.array([paramDict['kinematic_visc'], 
                         paramDict['ring_loc'],
                         paramDict['ring_mass'],
                         init_col/paramDict['col_ratio'],
                         paramDict['init_temp']*kB/(mu*mH)],
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)
    eInt = None
    tstart = 0.0
    tend = paramDict['end_time']*t0
    paramDict['dt_tol'] = 1.0
    dt = dt*t0

elif probname == 'gidisk' :

    vphi = paramDict['rot_curve_velocity']
    mdot = -paramDict['obc_pres_val']
    eta = paramDict['eta']
    chi = G*mdot/vphi**3
    s = 1.0/2.0**0.5 * (chi/eta)**(1./3.)
    torb = 2.0*np.pi*paramDict['rmax']/vphi
    h_steady = paramDict['gamma']/(paramDict['gamma']-1.0) * (s*vphi)**2
    paramDict['obc_enth_val'] = h_steady*paramDict['obc_vdisp']**2 \
                                / paramDict['init_col']
    paramDict['dt_start'] = paramDict['dt_init']*torb
    paramArr = np.array([eta, chi, paramDict['t_Q']], 
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)
    grd = vader.grid(paramDict)
    R = grd.r_h[-1]
    col1 = vphi**2 * (chi/eta)**(1./3.) / (np.pi*G*R)
    colSteady = col1 * (R/grd.r)
    presSteady = colSteady*(s*vphi)**2
    col = colSteady * paramDict['init_col']
    pres = presSteady * paramDict['init_col'] * paramDict['init_vdisp']**2
    tstart = 0.0
    tend = paramDict['n_orbit']*torb
    eInt = None
    dt = dt*torb

elif probname == 'ringrad' :

    r0 = paramDict['ring_loc']
    t0 = r0**2/(12.0*paramDict['kinematic_visc'])
    col0 = paramDict['ring_mass'] / (np.pi * r0**2)
    grd = vader.grid(paramDict)
    idx = np.argmax(grd.r_h > r0) - 1
    init_col = paramDict['ring_mass'] / grd.area[idx]
    col = np.ones(grd.nr)*init_col/paramDict['col_ratio']
    col[idx] = init_col
    fz0 = paramDict['f_z0']
    temp = paramDict['init_teff']
    pres = col*temp*kB/(mu*mH) + (1./3.)*fz0*a*temp**4
    gammaGas = 5./3.
    eInt = col*temp*kB/(mu*mH*(gammaGas-1)) + fz0*a*temp**4
    tstart = 0.0
    tend = paramDict['end_time']*t0
    paramArr = np.array([paramDict['kinematic_visc'], 
                         paramDict['ring_loc'],
                         paramDict['ring_mass'],
                         init_col/paramDict['col_ratio'],
                         paramDict['init_teff']*kB/(mu*mH)],
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)
    paramDict['dt_tol'] = 1.0
    dt = dt*t0

# Take a CN timestep
colTmp = np.copy(col)
presTmp = np.copy(pres)
if eInt is not None:
    eIntTmp = np.copy(eInt)
else:
    eIntTmp = None
outCN = vader.advance(tstart, dt, grd, colTmp, presTmp, paramDict,
                      method='CN', verbosity=0, c_params=paramPtr,
                      eInt = eIntTmp, err_tol=1e-10,
                      max_iter=max_iter)

# Take a BE time step
colTmp = np.copy(col)
presTmp = np.copy(pres)
if eInt is not None:
    eIntTmp = np.copy(eInt)
else:
    eIntTmp = None
outBE = vader.advance(tstart, dt, grd, colTmp, presTmp, paramDict,
                      method='BE', verbosity=0, c_params=paramPtr,
                      eInt = eIntTmp,
                      err_tol=1e-10, max_iter=max_iter)

# Now run the full simulation with CN and BE
colTmp = np.copy(col)
presTmp = np.copy(pres)
if eInt is not None:
    eIntTmp = np.copy(eInt)
else:
    eIntTmp = None
outfullCN = vader.driver(tstart, tend, grd, colTmp, presTmp, paramDict,
                         eInt=eIntTmp,
                         nOut=0, c_params=paramPtr, max_iter=max_iter1,
                         verbosity=1, method='CN', max_step=max_step)
colTmp = np.copy(col)
presTmp = np.copy(pres)
if eInt is not None:
    eIntTmp = np.copy(eInt)
else:
    eIntTmp = None
outfullBE = vader.driver(tstart, tend, grd, colTmp, presTmp, paramDict,
                         eInt=eIntTmp,
                         nOut=0, c_params=paramPtr, max_iter=max_iter1,
                         verbosity=1, method='BE', max_step=max_step)

# Create directoy into which data can be written
dirname = os.path.join(_path, 'output')
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Write data to file
fname = probname+'_'+aa_m+'_'+logdt+'.txt'
fullname = os.path.join(dirname, fname)
fp=open(fullname, 'w')
fp.write("{:f} {:f} {:f} {:f} {:f} {:f}\n".
         format(outCN.advanceTime, outCN.nextIterTime, outCN.userTime,
                outBE.advanceTime, outBE.nextIterTime, outBE.userTime))
fp.write("{:d} {:d}\n".format(outCN.nIter, outBE.nIter))
for i in range(max_iter):
    fp.write("{:e} {:d} {:e} {:d}\n".
             format(outCN.resid[i], outCN.rtype[i], 
                    outBE.resid[i], outBE.rtype[i]))
fp.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n".
         format(outfullCN.driverTime, outfullCN.advanceTime, 
                outfullCN.nextIterTime, outfullCN.userTime,
                outfullBE.driverTime, outfullBE.advanceTime,
                outfullBE.nextIterTime, outfullBE.userTime))
fp.write("{:e} {:d} {:d} {:d} {:e} {:d} {:d} {:d}\n".
         format(
             outfullCN.tFin,
             outfullCN.nStep, outfullCN.nFail,
             outfullCN.nIter,
             outfullBE.tFin,
             outfullBE.nStep, outfullBE.nFail,
             outfullBE.nIter
         ))
for i in range(max_iter1):
    fp.write("{:e} {:e}\n".
             format(outfullCN.residSum[i], outfullBE.residSum[i]))
for i in range(max_step):
    fp.write("{:d} {:d}\n".
             format(outfullCN.iterStep[i], outfullBE.iterStep[i]))
fp.close()


