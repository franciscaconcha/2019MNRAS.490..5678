"""
This function runs the Krumholz & Burkert (2010, ApJ, 724, 895)
gravitational instability-dominated disk problem.
"""

import numpy as np
import ctypes
import vader
from collections import namedtuple
import scipy.constants as physcons

G = physcons.G*1e3   # G in CGS units

def gidisk(paramFile, nSave=16, saveTimes=None, **kwargs):
    """
    This function runs the Krumholz & Burkert (2010, ApJ, 724, 895)
    gravitational instability-dominated disk problem.

    Parameters
    ----------
    paramFile : string
       Name of the parameter file to use for the run
    saveTimes : arraylike or None
       array or list of output times in sec. If None, this list is
       constructed the nSave parameter. The first element of saveTimes
       must equal tStart, and the last element must equal tEnd.
    nSave : int
       number of output times to store between initial and final
       time. This is ignored if saveTimes is set
    kwargs : optional
       Any keywords passed here will be added to the paramDict
       produced by reading the paramFile. For example, set nr = NN to
       change the resolution to NN, overriding what is in the file.

    Returns
    -------
    A namedtuple with the following elements:
    x : array, shape (nr)
       dimensionless cell center positions, in units where the
       outer radius of the disk is 1
    T : array, shape (nSave)
       output times, in units where the outer orbital time is 1
    colOut : array, shape (nSave, nr)
       column density at specified times, in units where the steady
       state solution is at the disk edge is 1
    presOut : array, shape (nSave, nr)
       vertically-integrated pressure at specified times, in units
       where the steady state solution for the column density times
       velocity dispersion squared at the disk edge is 1
    Q : array, shape (nSave, nr)
       Toomre Q parameter at specified tiems
    colSteady : array, shape (nr)
       steady state solution for column density at specified times, in
       the same dimensionless units as colOut
    presSteady :  array, shape (nr)
       steady state solution for vertically-integrate pressure at
       specified times, in the same dimensionless units as presOut
    nStep : long
       number of simulation time steps completed
    nIter : long
       total number of implicit iterations completed
    nFail : long
       total number of time steps where the implicit solver failed to
       converge
    """

    # Read parameter file, and apply any extra keywords
    paramDict = vader.readParam(paramFile)
    for k in kwargs:
        paramDict[k] = kwargs[k]

    # Compute characteristic dimensionless numbers
    vphi = paramDict['rot_curve_velocity']
    mdot = -paramDict['obc_pres_val']
    eta = paramDict['eta']
    chi = G*mdot/vphi**3
    s = 1.0/2.0**0.5 * (chi/eta)**(1./3.)
    torb = 2.0*np.pi*paramDict['rmax']/vphi

    # Set the enthalpies correctly
    h_steady = paramDict['gamma']/(paramDict['gamma']-1.0) * (s*vphi)**2
    paramDict['obc_enth_val'] = h_steady*paramDict['obc_vdisp']**2 \
                                / paramDict['init_col']

    # Set starting timestep
    paramDict['dt_start'] = paramDict['dt_init']*torb

    # Set up the parameters to pass to c
    paramArr = np.array([eta, chi, paramDict['t_Q']], 
                        dtype='d')
    paramPtr = ctypes.byref(paramArr.ctypes. \
                            data_as(ctypes.POINTER(ctypes.c_double)).contents)

    # Set up initial conditions and calculate steady state
    grd = vader.grid(paramDict)
    R = grd.r_h[-1]
    col1 = vphi**2 * (chi/eta)**(1./3.) / (np.pi*G*R)
    colSteady = col1 * (R/grd.r)
    presSteady = colSteady*(s*vphi)**2
    col = colSteady * paramDict['init_col']
    pres = presSteady * paramDict['init_col'] * paramDict['init_vdisp']**2

    # Run the simulation
    vaderOut = vader.vader(0, paramDict['n_orbit']*torb, paramDict,
                           grd=grd, col=col, pres=pres,
                           nSave=nSave, saveTimes=saveTimes,
                           c_params = paramPtr)
    tOut = vaderOut.tOut
    colOut = vaderOut.colOut
    presOut = vaderOut.presOut
    Q = np.sqrt(2.0*(grd.beta+1))*grd.vphi/grd.r * \
        np.sqrt(presOut/colOut) / (np.pi*G*colOut)

    # Construct the output object
    ret_obj = namedtuple('VADER_gidisk_out',
                         'x T colOut presOut Q colSteady presSteady' +
                         ' nStep nIter nFail')
    ret = ret_obj(grd.r/R, tOut/torb, colOut/col1, presOut/(col1*vphi**2), \
                  Q, colSteady/col1, presSteady/(col1*vphi**2),
                  vaderOut.nStep, vaderOut.nIter, vaderOut.nFail)

    # Return
    return ret
