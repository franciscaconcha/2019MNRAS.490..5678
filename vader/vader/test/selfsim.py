"""
This routine runs the self-similar disk evolution test, following the
analytic solution of Lynden-Bell & Pringle (1974, MNRAS, 168, 603).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import vader
from collections import namedtuple

# Physical constants in CGS units
import scipy.constants as physcons
kB = physcons.k*1e7
mH = 1e3*physcons.physical_constants['atomic mass constant'][0]*1.00794
mu = 2.33

def selfsim(paramFile, nSave=65, saveTimes=None, **kwargs):
    """
    This routine runs the self-similar disk evolution test, following
    the analytic solution of Lynden-Bell & Pringle (1974, MNRAS, 168,
    603).

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
       T : array, shape (nOut)
          output times in dimensionless units
       colOut : array, shape (nOut, nr)
          column density at times tOut and positions x in dimensionless
          units
       colExact : array, shape (nOut, nr)
          exact analytic solution for column density at same times and
          positions as colOut
       err : array, shape (nOut, nr)
          fractional error in numerical solution for column density
       presOut : array, shape (nOut, nr)
          pressures at same times, positions as colOut, in dimensionless
          units
       l1err : array, shape (nOut)
          L1 error in the numerical solution at specified times
       mBndOut : array, shape (nOut, 2)
          cumulative mass (in dimensionless units) transported across
          the inner and outer boundaries up to time tOut; positive
          values indicate transport in the +r direction
       mDiskOut : array, shape (nOut)
          total mass in the computational domain at time tOut
       eBndOut : array, shape (nOut, 2)
          cumulative energy (in dimensionless units) transported across
          the inner and outer boundaries up to time tOut; positive
          values indicate transport in the +r direction
       eDiskOut : array, shape (nOut)
          total energy in the computational domain at time tOut
       nStep : long
          total number of simulation steps
       nIter : long
          total number of implicit iterations completed
       nFail : long
          total number of failed convergences
    """

    # Read parameters
    paramDict = vader.readParam(paramFile)
    for k in kwargs:
        paramDict[k] = kwargs[k]

    # Compute the dimensionless time parameter
    R0 = paramDict['R0']
    ts = R0**2/(3*paramDict['nu0'])

    # Set up the intial conditions
    grd = vader.grid(paramDict)
    x = grd.r/R0
    col1 = paramDict['Mdot0']/(3*np.pi*paramDict['nu0'])
    col = col1 * np.exp(-x)/x
    pres = col*paramDict['init_temp']*kB/(mu*mH)
    pres1 = col1*paramDict['init_temp']*kB/(mu*mH)

    # Set up pointer to parameters to pass to c routines
    paramArr = np.array([paramDict['nu0'], paramDict['R0'],
                         paramDict['Mdot0'], 
                         paramDict['init_temp']*kB/(mu*mH)], 
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).
                            contents)

    # Run the simulation
    vaderOut = vader.vader(ts, paramDict['end_time']*ts, 
                           paramDict, col=col, pres=pres, grd=grd,
                           nSave=nSave, saveTimes=saveTimes, 
                           c_params=paramPtr)
    tOut = vaderOut.tOut
    colOut = vaderOut.colOut
    presOut = vaderOut.presOut
    mBndOut = vaderOut.mBndOut
    eBndOut = vaderOut.eBndOut
    mDiskOut = np.zeros(len(tOut))
    eDiskOut = np.zeros(len(tOut))
    for i in range(len(tOut)):
        mDiskOut[i] = np.sum(colOut[i,:]*grd.area)
        eDiskOut[i] \
            = np.sum( (presOut[i,:]/(paramDict['gamma']-1) +
                       colOut[i,:]*grd.psiEff) *
                      grd.area )

    # Generate exact analytic solution
    T = tOut/ts
    colExact = np.zeros((len(tOut), grd.nr))
    for i, t in enumerate(T):
        colExact[i,:] = col1 * np.exp(-x/t) / (x*t**1.5)
    err = (colOut-colExact) / colExact

    # Compute L1 error
    diff = np.abs(colOut - colExact)
    l1err = np.sum(np.abs(grd.area*diff), axis=1)/(col1*np.pi*R0**2)

    # Put output quantities in a namedtuple
    ret_obj = namedtuple('VADER_selfsim_out',
                         'x T colOut colExact err presOut l1err '
                         + 'mBndOut mDiskOut eBndOut eDiskOut '
                         + 'nStep nIter nFail')
    ret = ret_obj(x, T, colOut/col1, colExact/col1, err,
                  presOut/pres1, l1err, mBndOut/(col1*R0**2),
                  mDiskOut/(col1*R0**2),
                  eBndOut/(pres1*R0**2)*(paramDict['gamma']-1),
                  eDiskOut/(pres1*R0**2)*(paramDict['gamma']-1),
                  vaderOut.nStep, vaderOut.nIter, vaderOut.nFail)

    # Return
    return ret

