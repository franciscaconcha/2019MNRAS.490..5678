"""
This module provides a function to read vader checkpoints.
"""

# List of routines provided
__all__ = ['readCheckpoint']

import numpy as np
from collections import namedtuple
import struct
from .grid import grid

# Checkpoint reading routine
def readCheckpoint(chk, basic_only=True):
    """
    Reads a vader checkpoint

    Parameters:
       chk : string | file
          Name or handle of file to be read
       basic_only : bool
          If True, neither the constraint vector nor the user data
          buffer are read; these will be returned as None

    Returns:
    out : nametuple
       out is a namedtuple containing the following output items:

       chkNum : int
          sequential number of this checkpoint file
       t : float
          current simulation time
       dt : float
          current simulation time step
       tOut : array, shape (nOut)
          output times
       colOut : array, shape (nOut,nr)
          2D array of column densities stored at specified times
       presOut : array, shape (nOut,nr)
          2D array of pressures stored at specified times
       eIntOut : array, shape (nOut,nr)
          2D array of internal energies stored at specified times;
          returned only if present in checkpoint file, otherwise set
          to None
       mBndOut : array, shape (nOut,2)
          cumulative mass transported across the inner and outer
          boundaries up to the specified time; positive values
          indicate transport in the +r direction, negative values
          indicate transport in the -r direction
       eBndOut : array, shape (nOut,2)
          cumulateive energy transported across the inner and outer
          boundaries up to the specified time; positive values
          indicate transport in the +r direction, negative values
          indicate transport in the -r direction
       mSrcOut : array, shape (nOut,nr)
          cumulative mass column density added to each cell by the
          source terms up to the specified time; returned only if
          present in the checkpoint file, otherwise set to None
       eSrcOut : array, shape (nOut,nr)
          cumulative energy per unit area added to each cell by the
          source terms up to the specified time; returned only if
          present in the checkpoint file, otherwise set to None
       userOut : array, shape (nOut, nUserOut, nr)
          user-defined outputs, returned only if present in the
          checkpoint file, otherwise set to None
       nStep : long
          total number of simulation timesteps taken up to this
          checkpoint
       nIter : long
          total number of implicit iterations, summed over all time
          steps
       nFail : long
          total number of times the implicit solver failed to converge
       constraint : array
          the constraint vector used for Anderson acceleration; will
          be None if calculation was run without Anderson
          acceleration
       buf : string
          a buffer containing any extra, user-defined data that lives
          in the checkpoint; it is up to the user to process this
    """

    # If we were given a string, open the file
    if type(chk) is not str:
        fp = chk
    else:
        fp = open(chk, 'rb')

    # Read control information
    data = fp.read(struct.calcsize('L'))
    chkNum, = struct.unpack('L', data)
    data = fp.read(struct.calcsize('?'))
    eos, = struct.unpack('?', data)
    data = fp.read(struct.calcsize('?'))
    massSrc, = struct.unpack('?', data)
    data = fp.read(struct.calcsize('?'))
    intEnSrc, = struct.unpack('?', data)
    data = fp.read(struct.calcsize('L'))
    nUserOut, = struct.unpack('L', data)

    # Read the time information
    data = fp.read(struct.calcsize('d'))
    t, = struct.unpack('d', data)
    data = fp.read(struct.calcsize('d'))
    dt, = struct.unpack('d', data)
    data = fp.read(struct.calcsize('L'))
    nStep, = struct.unpack('L', data)
    data = fp.read(struct.calcsize('L'))
    nIter, = struct.unpack('L', data)
    data = fp.read(struct.calcsize('L'))
    nFail, = struct.unpack('L', data)

    # Read the grid
    grd = grid(None, chk=fp)

    # Read number of outputs in the file
    data = fp.read(struct.calcsize('L'))
    nOut, = struct.unpack('L', data)

    # Read the output times
    data = fp.read(struct.calcsize('d'*nOut))
    tOut = np.array(struct.unpack('d'*nOut, data))

    # Read the output data
    data = fp.read(struct.calcsize('d'*nOut*grd.nr))
    colOut = np.array(struct.unpack('d'*nOut*grd.nr, data)). \
             reshape((nOut, grd.nr))
    data = fp.read(struct.calcsize('d'*nOut*grd.nr))
    presOut = np.array(struct.unpack('d'*nOut*grd.nr, data)). \
              reshape((nOut, grd.nr))
    if eos:
        data = fp.read(struct.calcsize('d'*nOut*grd.nr))
        eIntOut = np.array(struct.unpack('d'*nOut*grd.nr, data)). \
                  reshape((nOut, grd.nr))
    else:
        eIntOut = None
    data = fp.read(struct.calcsize('d'*nOut*2))    
    mBndOut = np.array(struct.unpack('d'*nOut*2, data)). \
              reshape(nOut, 2)
    data = fp.read(struct.calcsize('d'*nOut*2))    
    eBndOut = np.array(struct.unpack('d'*nOut*2, data)). \
              reshape(nOut, 2)
    if massSrc:
        data = fp.read(struct.calcsize('d'*nOut*grd.nr))
        mSrcOut = np.array(struct.unpack('d'*nOut*grd.nr, data)). \
                  reshape((nOut, grd.nr))
    else:
        mSrcOut = None
    if massSrc or intEnSrc:
        data = fp.read(struct.calcsize('d'*nOut*grd.nr))
        eSrcOut = np.array(struct.unpack('d'*nOut*grd.nr, data)). \
                  reshape((nOut, grd.nr))
    else:
        eSrcOut = None

    # Read user outputs
    if nUserOut > 0:
        data = fp.read(struct.calcsize('d'*
                                       nOut*grd.nr*nUserOut))
        userOut = np.array(
            struct.unpack('d'*nOut*nUserOut*grd.nr, data)). \
            reshape((nOut, nUserOut, grd.nr))
    else:
        userOut = None

    # Non-basic stuff
    if not basic_only:
        
        # Read Anderson acceleration parameter
        aa_m = struct.unpack('i', fp.read(struct.calcsize('i')))[0]

        # Read constraint vector
        if aa_m > 0:
            data = fp.read(struct.calcsize('d'*(3*grd.nr+1)))
            constraint = np.array(struct.unpack('d'*(3*grd.nr+1), data))
        else:
            constraint = None

        # Read any remaining data into a buffer
        buf = fp.read()

    else:

        # Don't read non-basic stuff
        aa_m = None
        constraint = None
        buf = None

    # Close the file if we opened it
    if type(chk) is str:
        fp.close()
        
    # Construct the object to return
    ret_obj = namedtuple('VADER_out', 
                         'grid chkNum t dt tOut colOut presOut eIntOut '+
                         'mBndOut eBndOut mSrcOut eSrcOut '+
                         'userOut nStep nIter nFail aa_m constraint buf')
    ret = ret_obj(grd, chkNum, t, dt, tOut, colOut, presOut, eIntOut,
                  mBndOut, eBndOut, mSrcOut, eSrcOut,
                  userOut, nStep, nIter, nFail, aa_m, constraint, buf)
    return ret
