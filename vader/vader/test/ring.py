"""
This function runs the viscous ring evolution test, following the
analytic solution from Pringle (1981, ARAA, 19, 137).
"""

import numpy as np
from scipy.special import ive
import ctypes
import vader
from collections import namedtuple

# Physical constants in CGS units
import scipy.constants as physcons
kB = physcons.k*1e7
mH = 1e3*physcons.physical_constants['atomic mass constant'][0]*1.00794
mu = 2.33

def ring(paramFile, nSave=65, saveTimes=None, checkpoint=None, **kwargs):
    """
    This function runs the viscous ring evolution test, following the
    analytic solution from Pringle (1981, ARAA, 19, 137).

    Parameters
    ----------
    paramFile : string
       Name of the parameter file to use for the run
    nSave : int
       number of output times to store between initial and final
       time. This is ignored if outTimes is set
    saveTimes : arraylike or None
       array or list of output times in sec. If None, this list is
       constructed from the nSave parameter. The first element of
       saveTimes must equal tStart, and the last element must equal
       tEnd.
    checkpoint : str
       if not None, outputs are written out as checkpoints to disk, in
       files whose names follow the pattern checkpoint_NNNN.vader
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
       tOut : array, shape (nOut)
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
          cumulative mass advected in positive direction acros inner,
          outer boundaries up to time tOut, in dimensionless units
       mDiskOut : array, shape (nOut)
          instantaneous mass in the simulation domain at times tOut, in
          dimensionless units
       eBndOut : array, shape (nOut, 2)
          cumulative energy advected in positive direction acros inner,
          outer boundaries up to time tOut, in dimensionless units
       eDiskOut : array, shape (nOut)
          instantaneous energy in the simulation domain at times tOut, in
          dimensionless units
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
    pres0 = col0 * paramDict['init_temp']*kB/(mu*mH)

    # Set up the intial conditions
    grd = vader.grid(paramDict)
    idx = np.argmax(grd.r_h > r0) - 1
    init_col = paramDict['ring_mass'] / grd.area[idx]
    col = np.ones(grd.nr)*init_col/paramDict['col_ratio']
    col[idx] = init_col
    pres = col*paramDict['init_temp']*kB/(mu*mH)

    # Set up parameter array to pass to c routines
    paramArr = np.array([paramDict['kinematic_visc'], 
                         paramDict['ring_loc'],
                         paramDict['ring_mass'],
                         init_col/paramDict['col_ratio'],
                         paramDict['init_temp']*kB/(mu*mH)],
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes.
                            data_as(ctypes.POINTER(ctypes.c_double)).
                            contents)

    # Set up checkpointing
    if checkpoint is not None:
        chk = True
    else:
        chk = False
    
    # Run simulations
    vaderOut = vader.vader(
        0, paramDict['end_time']*t0, paramDict,
        grd=grd, col=col, pres=pres, 
        nSave=nSave, saveTimes=saveTimes,
        checkpoint=chk, checkname=checkpoint,
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

    # Generate corresponding exact analytic solutions
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

    # Compute L1 error
    diff = np.abs(colOut - colExact)
    diff[colExact < init_col/paramDict['col_ratio']**0.5] = 0.0
    l1err = np.sum(np.abs(grd.area*diff), axis=1)/paramDict['ring_mass']

    # Put output quantities in a namedtuple
    ret_obj = namedtuple('VADER_ring_out',
                         'x tau colOut colExact err presOut l1err '
                         + 'mBndOut mDiskOut eBndOut eDiskOut '
                         + 'nStep nIter nFail')
    ret = ret_obj(x, tau, colOut/col0, colExact/col0, err, presOut/pres0,
                  l1err,  mBndOut/(col0*r0**2), mDiskOut/(col0*r0**2), 
                  eBndOut/(pres0*r0**2)*(paramDict['gamma']-1), 
                  eDiskOut/(pres0*r0**2)*(paramDict['gamma']-1),
                  vaderOut.nStep, vaderOut.nIter, vaderOut.nFail)

    # Return
    return ret

