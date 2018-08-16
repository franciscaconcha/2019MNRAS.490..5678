"""
This function runs the viscous ring evolution test, following the
analytic solution from Pringle (1981, ARAA, 19, 137).
"""

import numpy as np
from scipy.special import ive
from scipy.optimize import brentq
import ctypes
import vader
from collections import namedtuple

# Physical constants in CGS units
import scipy.constants as physcons
kB = physcons.k*1e7
mH = 1e3*physcons.physical_constants['atomic mass constant'][0]*1.00794
mu = 0.61
sigma = physcons.sigma*1e3
c = physcons.c*1e2
a = 4*sigma/c

def presResid(logt, col, pres, fz0):
    return pres - kB/(mu*mH)*col*np.exp(logt) \
        - (1./3.)*a*fz0*np.exp(logt)**4

def tempFromPres(col, pres, fz0, tmin=1e-30, tmax=1e30):
    """
    This routine takes as arguments a set of column density and
    pressure arrays, and a value for the scale height factor f_z0, and
    returns the corresponding temperatures.

    Parameters
    ----------
    col : array
       column densities (cgs units)
    pres : array
       vertically-integrated pressures (cgs units)
    f_z0 : float
       scale height factor (cgs units)
    tmax : float
       lower bound on temperature (must be > 0); used a bound in
       Brent's method
    tmax : float
       upper bound on temperature; used a bound in Brent's method

    Returns
    -------
    temp : array
       temperatures in K
    """

    logt = 0.0*col
    logtmin = np.log(tmin)
    logtmax = np.log(tmax)
    for idx, dummy in np.ndenumerate(logt):
        logt[idx] = brentq(presResid, logtmin, logtmax, 
                            args=(col[idx], pres[idx], fz0))
    return np.exp(logt)


def ringrad(paramFile, nSave=65, saveTimes=None, **kwargs):
    """
    This function runs the viscous ring evolution test, following the
    analytic solution from Pringle (1981, ARAA, 19, 137). The equation
    of state used to evolve the internal energy is one that includes both
    radiation pressure and gas pressure effects.

    Parameters
    ----------
    paramFile : string
       Name of the parameter file to use for the run
    nSave : int
       number of output times to store between initial and final
       time. This is ignored if saveTimes is set
    saveTimes : arraylike or None
       array or list of output times in sec. If None, this list is
       constructed the nSave parameter. The first element of saveTimes
       must equal tStart, and the last element must equal tEnd.
    kwargs : optional
       Any keywords passed here will be added to the paramDict
       produced by reading the paramFile. For example, set nr = NN to
       change the resolution to NN, overriding what is in the file.

    Returns
    -------
    out : namedtuple
       out is a namedtuple containing the following output items:
       x : array, shape (nr)
          dimensionless cell centers positions
       tOut : array, shape (nSave)
          output times in dimensionless units
       colOut : array, shape (nSave, nr)
          column density at times tOut and positions x
       colExact : array, shape (nSave, nr)
          exact analytic solution for column density at same times and
          positions as colOut
       err : array, shape (nSave, nr)
          fractional error in numerical solution for column density
       presOut : array, shape (nSave, nr)
          pressures at same times, positions as colOut
       eIntOut : array, shape (nSave, nr)
          internal energies at same times, positions as colOut
       tempOut : array, shape (nSave, nr)
          temperatures at same times, positions as colOut
       l1err : array, shape (nSave)
          L1 error in the numerical solution at specified times
       nStep : long
          total number of simulation steps
       nIter : long
          total number of implicit iterations completed
       nFail : long
          total number of failed convergences
    """

    # Read parameter file, and apply any extra keywords
    paramDict = vader.readParam(paramFile)
    for k in kwargs:
        paramDict[k] = kwargs[k]
 
    # Compute characteristic non-dimensional numbers
    r0 = paramDict['ring_loc']
    t0 = r0**2/(12.0*paramDict['kinematic_visc'])
    col0 = paramDict['ring_mass'] / (np.pi * r0**2)

    # Set up the intial conditions
    grd = vader.grid(paramDict)
    idx = np.argmax(grd.r_h > r0) - 1
    init_col = paramDict['ring_mass'] / grd.area[idx]
    col = np.ones(grd.nr)*init_col/paramDict['col_ratio']
    col[idx] = init_col
    fz0 = paramDict['f_z0']
    temp = paramDict['init_teff']
    pres = col*temp*kB/(mu*mH) + (1./3.)*fz0*a*temp**4
    if paramDict['gamma'] == 'c_func':
        gammaGas = 5./3.
        eInt = col*temp*kB/(mu*mH*(gammaGas-1)) + fz0*a*temp**4
    else:
        eInt = None

    # Set up parameter array to pass to c routines
    paramArr = np.array([paramDict['kinematic_visc'], 
                         paramDict['ring_loc'],
                         paramDict['ring_mass'],
                         init_col/paramDict['col_ratio'],
                         paramDict['init_teff']*kB/(mu*mH)],
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)

    # Run simulations
    vaderOut = vader.vader(0, paramDict['end_time']*t0, paramDict,
                           grd=grd, col=col, pres=pres, eInt=eInt,
                           nSave=nSave, saveTimes=saveTimes,
                           c_params=paramPtr)

    # Grab results nad compute some derived quantities
    tOut = vaderOut.tOut
    colOut = vaderOut.colOut
    presOut = vaderOut.presOut
    eIntOut = vaderOut.eIntOut
    if eIntOut is None:
        tempOut = presOut/colOut * mu*mH/kB
        eIntOut = presOut / (paramDict['gamma']-1)
        pRadOut = None
        pGasOut = presOut
    else:
        tempOut = tempFromPres(colOut, presOut, fz0)
        pRadOut = (1./3.)*fz0*a*tempOut**4
        pGasOut = colOut*tempOut*kB/(mu*mH)
    mBndOut = vaderOut.mBndOut
    eBndOut = vaderOut.eBndOut
    eGravOut = 2.0*colOut*grd.psiEff
    eOrbOut = -colOut*grd.psiEff
    eOut = eIntOut + eGravOut + eOrbOut
    mDiskOut = np.zeros(len(tOut))
    eDiskOut = np.zeros(len(tOut))
    for i in range(len(tOut)):
        mDiskOut[i] = np.sum(colOut[i,:]*grd.area)
        eDiskOut[i] = np.sum(eOut[i,:]*grd.area)

    # Generate exact analytic solutions for column density
    tau = tOut/t0
    tau[0]=1e-30   # Avoid complaints about divide-by-zero
    x = grd.r/r0
    colExact = np.zeros((len(tau), grd.nr))
    for i, t in enumerate(tau):
        colExact[i,:] = col0 * (1/t) * \
                         x**(-0.25) * np.exp(-(1-x)**2/t) \
                         * ive(0.25, 2*x/t)
    colExact[np.logical_not(np.isfinite(colExact))] = 0.0
    err = (colOut-colExact-init_col/paramDict['col_ratio']) / \
          (colExact + init_col/paramDict['col_ratio']**0.5)

    # Construct return object
    ret_obj = namedtuple('VADER_ringrad_out',
                         'x tau colOut colExact err presOut ' +
                         'pGasOut pRadOut eIntOut eGravOut ' +
                         'eOrbOut eOut tempOut eDiskOut ' +
                         'eBndOut nStep nIter nFail')
    ret = ret_obj(x, tau, colOut, colExact, err, presOut, 
                  pGasOut, pRadOut, eIntOut, eGravOut, eOrbOut, 
                  eOut, tempOut, eDiskOut, eBndOut,
                  vaderOut.nStep, vaderOut.nIter, vaderOut.nFail)

    # Return
    return ret
