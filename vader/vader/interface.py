"""
This module provides an interface to the vader c library.
"""

# List of routines provided
__all__ = ['gridInit', 'gridInitKeplerian', 'gridInitFlat', 
           'wkspAlloc', 'gridFree', 'wkspFree', 'rotCurveSpline', 
           'cDict', 'driver', 'advance', 'vader', 'loadlib']


from vader import grid
import numpy as np
import numpy.ctypeslib as npct
import os
from copy import deepcopy
from collections import namedtuple
from warnings import warn
from ctypes import POINTER, c_int, c_bool, c_ulong, c_double, \
    c_void_p, c_char_p, byref, Structure, pointer

##################################################################
# Define some types for use later                                #
##################################################################
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1,
                                 flags="CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2,
                                 flags="CONTIGUOUS")
array_3d_double = npct.ndpointer(dtype=np.double, ndim=3,
                                 flags="CONTIGUOUS")
array_1d_ulong = npct.ndpointer(dtype=c_ulong, ndim=1,
                                flags="CONTIGUOUS")
array_1d_bool = npct.ndpointer(dtype=c_bool, ndim=1,
                               flags="CONTIGUOUS")
c_enum = c_int      # c enums are always ints
c_double_p = POINTER(c_double)
c_double_pp = POINTER(c_double_p)
c_void_pp = POINTER(c_void_p)
c_bool_p = POINTER(c_bool)
c_ulong_p = POINTER(c_ulong)

##################################################################
# Define the c version of the grid structure                     #
##################################################################
class c_grid(Structure):
    _fields_ = [("nr", c_ulong),
                ("linear", c_bool),
                ("r_g", c_double_p),
                ("r_h", c_double_p),
                ("dr_g", c_double_p),
                ("area", c_double_p),
                ("vphi_g", c_double_p),
                ("vphi_h", c_double_p),
                ("beta_g", c_double_p),
                ("beta_h", c_double_p),
                ("psiEff_g", c_double_p),
                ("psiEff_h", c_double_p),
                ("g_h", c_double_p)]
c_grid_p = POINTER(c_grid)
c_grid_pp = POINTER(c_grid_p)

##################################################################
# Persistent variables describing the state of the library       #
##################################################################
__libvader = None
__testing_mode = False

##################################################################
# Default parameters for c functions                             #
##################################################################
__defDict = {
    'dt_start' : -1.0,
    'delta' : 0.0,
    'mass_src' : 0.0,
    'int_en_src' : 0.0,
    'err_tol' : 1.0e-6,
    'damp_factor' : 0.0,
    'dt_tol' : 0.1,
    'max_iter' : 40,
    'interp_order' : 1,
    'max_dt_increase' : 1.5,
    'dt_min' : 1.0e-15,
    'max_step' : -1,
    'method' : 'CN',
    'pre_timestep' : None,
    'post_timestep' : None,
    'user_check_read' : None,
    'user_check_write' : None,
    'verbosity' : 0
}

##################################################################
# Routine to load the c library                                  #
##################################################################
def loadlib(path=None):
    """
    This function attempts to load the c library and define and
    interface to its routines.

    Parameters
    ----------
    path : string
       The path where the library is located. Defaults to
       ../vader_csrc, evaluated relative to the location of this
       file.

    Returns
    -------
    Nothing

    Throws
    ------
    If the library cannot be loaded, throws ImportError
    """

    # Refer to the global quantities
    global __libvader, __testing_mode

    # Do nothing if library is already loaded
    if __libvader is not None:
        return

    # Set path if not specified
    if path is None:
        path = os.path.join(os.path.join(os.path.dirname(__file__),
                                         '..'), 'bin')
    # Load library
    __libvader = npct.load_library("libvader", os.path.realpath(path))
    if __libvader is None:
        raise ImportError("unable to load libvader")

    # First define an interface to the function that lets us know
    # whether we are in testing mode or not; then use it to determine
    # whether we are in testing mode
    __libvader.testingMode.restype = c_int
    __libvader.testingMode.argtype = []
    if __libvader.testingMode() == 1:
        __testing_mode = True
    else:
        __testing_mode = False

    # Interfaces to the allocation routines
    __libvader.gridInit.restype = c_grid_p
    __libvader.gridInit.argtypes \
        = [ c_ulong,         # nr
            array_1d_double, # r_g
            array_1d_double, # r_h
            array_1d_double, # vphi_g
            array_1d_double, # vphi_h
            array_1d_double, # beta_g
            array_1d_double, # beta_h
            array_1d_double, # psiEff_g
            array_1d_double, # psiEff_h
            array_1d_double, # g_h
            c_bool           # linear
       ]
    __libvader.gridInitKeplerian.restype = c_grid_p
    __libvader.gridInitKeplerian.argtypes \
        = [ c_ulong,         # nr
            c_double,        # r_min
            c_double,        # r_max
            c_double,        # mass
            c_bool           # linear
       ]
    __libvader.gridInitFlat.restype = c_void_p
    __libvader.gridInitFlat.argtypes \
        = [ c_ulong,         # nr
            c_double,        # r_min
            c_double,        # r_max
            c_double,        # vphi
            c_bool           # linear
       ]
    __libvader.wkspAlloc.restype = c_void_p
    __libvader.wkspAlloc.argtypes = [c_ulong]
    __libvader.gridFree.restype = None
    __libvader.gridFree.argtypes = [c_grid_p]
    __libvader.wkspFree.restype = None
    __libvader.wkspFree.argtypes = [c_void_p]
    __libvader.outputAlloc.restype = c_bool
    __libvader.outputAlloc.argtypes \
        = [ c_ulong,         # nOut
            c_bool,          # eos_func
            c_bool,          # massSrc_func
            c_bool,          # intEnSrc_func
            c_ulong,         # nUserOut
            c_grid_p,        # grd
            c_double_pp,     # tOut
            c_double_pp,     # colOut
            c_double_pp,     # presOut
            c_double_pp,     # eIntOut
            c_double_pp,     # mBndOut
            c_double_pp,     # eBndOut
            c_double_pp,     # mSrcOut
            c_double_pp,     # eSrcOut
            c_double_pp      # userOut
        ]
    __libvader.outputFree.restype = None
    __libvader.outputFree.argtypes \
        = [ c_double_pp,     # tOut
            c_double_pp,     # colOut
            c_double_pp,     # presOut
            c_double_pp,     # eIntOut
            c_double_pp,     # mBndOut
            c_double_pp,     # eBndOut
            c_double_pp,     # mSrcOut
            c_double_pp,     # eSrcOut
            c_double_pp      # userOut
        ]
    __libvader.outputResize.restype = c_bool
    __libvader.outputResize.argtypes \
        = [ c_ulong,         # nOut
            c_bool,          # eos_func
            c_bool,          # massSrc_func
            c_bool,          # intEnSrc_func
            c_ulong,         # nUserOut
            c_grid_p,        # grd
            c_double_pp,     # tOut
            c_double_pp,     # colOut
            c_double_pp,     # presOut
            c_double_pp,     # eIntOut
            c_double_pp,     # mBndOut
            c_double_pp,     # eBndOut
            c_double_pp,     # mSrcOut
            c_double_pp,     # eSrcOut
            c_double_pp      # userOut
        ]
    
    # Interface to the rotation curve spline fitter
    __libvader.rotCurveSpline.restype = None
    __libvader.rotCurveSpline.argtypes \
        = [ array_1d_double, # rTab
            array_1d_double, # vTab
            c_ulong,         # nTab
            c_ulong,         # bspline_degree
            c_ulong,         # bspline_breakpoints
            array_1d_double, # r
            c_ulong,         # nr
            array_1d_double, # vphi 
            array_1d_double, # psi
            array_1d_double  # beta
        ]

    # Interface to the c simulation driver
    __libvader.driver.restype = c_double
    if __testing_mode:
        __libvader.driver.argtypes \
            = [ 
                # Time and grid parameters
                c_double,        # tStart
                c_double,        # tEnd
                # Equation of state parameters
                c_bool,          # eos_func
                c_double,        # gamma_val
                c_double,        # delta_val
                # Dimensionless viscosity
                c_bool,          # alpha_func
                c_double,        # alpha_val
                # Inner boundary condition
                c_enum,          # ibc_pres
                c_enum,          # ibc_enth
                c_bool,          # ibc_func
                c_double,        # ibc_pres_val
                c_double,        # ibc_enth_val
                # Outer boundary condition
                c_enum,          # obc_pres
                c_enum,          # obc_enth
                c_bool,          # obc_func
                c_double,        # obc_pres_val
                c_double,        # obc_enth_val
                # Source terms
                c_bool,          # massSrc_func
                c_double,        # massSrc_val
                c_bool,          # intEnSrc_func
                c_double,        # intEnSrc_val
                # Control parameters
                c_double,        # dtStart
                c_double,        # dtMin
                c_double,        # dtTol
                c_double,        # errTol
                c_double,        # maxDtIncrease
                c_ulong,         # maxIter
                c_ulong,         # interpOrder
                c_ulong,         # maxStep
                c_bool,          # useBE
                c_bool,          # preTimestep_func
                c_bool,          # postTimestep_func
                c_ulong,         # verbosity
                # Output control parameters; note that we declare
                # these as pointers intead of arrays because they can
                # be None / NULL
                c_ulong,         # nSave
                c_double_p,      # tSave
                c_ulong,         # nUserOut
                c_bool_p,        # userOutCum
                c_bool_p,        # writeCheckpoint
                c_char_p,        # checkname
                c_bool,          # userWriteCheckpoint
                c_bool,          # writeFirstStep
                c_ulong,         # checknum
                # Grid and workspace
                c_grid_p,        # grid
                c_void_p,        # wksp
                # Initial conditions
                array_1d_double, # col
                array_1d_double, # pres
                array_1d_double, # eInt
                c_void_p,        # params
                # Diagnostic outputs
                c_ulong_p,       # nStep
                c_ulong_p,       # nIter
                c_ulong_p,       # nFail
                # Storage for outputs
                c_ulong_p,       # nOut
                c_double_p,      # tOut
                c_double_p,      # colOut
                c_double_p,      # presOut
                c_double_p,      # eIntOut
                c_double_p,      # mBndOut
                c_double_p,      # eBndOut
                c_double_p,      # mSrcOut
                c_double_p,      # eSrcOut
                c_double_p,      # userOut
                # Testing mode parameters
                array_1d_double, # residSum
                array_1d_ulong,  # iterStep
                c_double_p,      # driverTime
                c_double_p,      # advanceTime
                c_double_p,      # nextIterTime
                c_double_p       # userTime
            ]
    else:
        __libvader.driver.argtypes \
            = [ 
                # Time and grid parameters
                c_double,        # tStart
                c_double,        # tEnd
                # Equation of state parameters
                c_bool,          # eos_func
                c_double,        # gamma_val
                c_double,        # delta_val
                # Dimensionless viscosity
                c_bool,          # alpha_func
                c_double,        # alpha_val
                # Inner boundary condition
                c_enum,          # ibc_pres
                c_enum,          # ibc_enth
                c_bool,          # ibc_func
                c_double,        # ibc_pres_val
                c_double,        # ibc_enth_val
                # Outer boundary condition
                c_enum,          # obc_pres
                c_enum,          # obc_enth
                c_bool,          # obc_func
                c_double,        # obc_pres_val
                c_double,        # obc_enth_val
                # Source terms
                c_bool,          # massSrc_func
                c_double,        # massSrc_val
                c_bool,          # intEnSrc_func
                c_double,        # intEnSrc_val
                # Control parameters
                c_double,        # dtStart
                c_double,        # dtMin
                c_double,        # dtTol
                c_double,        # errTol
                c_double,        # maxDtIncrease
                c_ulong,         # maxIter
                c_ulong,         # interpOrder
                c_ulong,         # maxStep
                c_bool,          # useBE
                c_bool,          # preTimestep_func
                c_bool,          # postTimestep_func
                c_ulong,         # verbosity
                # Output control parameters; note that we declare
                # these as pointers intead of arrays because they can
                # be None / NULL
                c_ulong,         # nSave
                c_double_p,      # tSave
                c_ulong,         # nUserOut
                c_bool_p,        # userOutCum
                c_bool_p,        # writeCheckpoint
                c_char_p,        # checkname
                c_bool,          # userWriteCheckpoint
                c_bool,          # writeFirstStep
                c_ulong,         # checknum
                # Grid and workspace
                c_grid_p,        # grid
                c_void_p,        # wksp
                # Initial conditions
                array_1d_double, # col
                array_1d_double, # pres
                array_1d_double, # eInt
                c_void_p,        # params
                # Diagnostic outputs
                c_ulong_p,       # nStep
                c_ulong_p,       # nIter
                c_ulong_p,       # nFail
                # Storage for outputs
                c_ulong_p,       # nOut
                c_double_p,      # tOut
                c_double_p,      # colOut
                c_double_p,      # presOut
                c_double_p,      # eIntOut
                c_double_p,      # mBndOut
                c_double_p,      # eBndOut
                c_double_p,      # mSrcOut
                c_double_p,      # eSrcOut
                c_double_p       # userOut
            ]

    # Interface to c advance routines
    __libvader.advanceCN.restype = c_double
    __libvader.advanceBE.restype = c_double
    if __testing_mode:
        __libvader.advanceCN.argtypes \
            = [ 
                # Time step and grid parameters
                c_double,        # t
                c_double,        # dt
                c_grid_p,        # grid
                # Starting data
                array_1d_double, # col
                array_1d_double, # pres
                array_1d_double, # eInt
                array_1d_double, # mBnd
                array_1d_double, # eBnd
                array_1d_double, # mSrc
                array_1d_double, # eSrc
                # Equation of state parameters
                c_bool,          # eos_func
                c_double,        # gamma_val
                c_double,        # delta_val
                # Dimensionless viscosity
                c_bool,          # alpha_func
                c_double,        # alpha_val
                # Inner boundary condition
                c_enum,          # ibc_pres
                c_enum,          # ibc_enth
                c_bool,          # ibc_func
                c_double,        # ibc_pres_val
                c_double,        # ibc_enth_val
                # Outer boundary condition
                c_enum,          # obc_pres
                c_enum,          # obc_enth
                c_bool,          # obc_func
                c_double,        # obc_pres_val
                c_double,        # obc_enth_val
                # Source terms
                c_bool,          # massSrc_func
                c_double,        # massSrc_val
                c_bool,          # intEnSrc_func
                c_double,        # intEnSrc_val
                # Control parameters
                c_double,        # errTol
                c_double,        # dtTol
                c_bool,          # maxIter
                c_ulong,         # interpOrder
                c_bool,          # noUpdate
                c_bool,          # verbose
                c_void_p,        # wksp
                c_void_p,        # params
                # Diagnostic outputs
                c_ulong_p,       # itCount
                array_1d_double, # resid
                array_1d_ulong,  # rType
                c_double_p,      # advanceTime
                c_double_p,      # nextIterTime
                c_double_p       # userTime
            ]
        __libvader.advanceBE.argtypes \
            = [ 
                # Time step and grid parameters
                c_double, #t
                c_double, #dt
                c_void_p, #grid
                # Starting data
                array_1d_double, #col
                array_1d_double, #pres
                array_1d_double, #eInt
                array_1d_double, # mBnd
                array_1d_double, # eBnd
                array_1d_double, # mSrc
                array_1d_double, # eSrc
                # Equation of state parameters
                c_bool, # eos_func
                c_double, # gamma_val
                c_double, # delta_val
                # Dimensionless viscosity
                c_bool, # alpha_func
                c_double, # alpha_val
                # Inner boundary condition
                c_enum, # ibc_pres
                c_enum, # ibc_enth
                c_bool, # ibc_func
                c_double, # ibc_pres_val
                c_double, # ibc_enth_val
                # Outer boundary condition
                c_enum, # obc_pres
                c_enum, # obc_enth
                c_bool, # obc_func
                c_double, # obc_pres_val
                c_double, # obc_enth_val
                # Source terms
                c_bool, # massSrc_func
                c_double, # massSrc_val
                c_bool, # intEnSrc_func
                c_double, # intEnSrc_val
                # Control parameters
                c_double, # errTol
                c_double, # dtTol
                c_bool, # maxIter
                c_ulong, # interpOrder
                c_bool, # noUpdate
                c_bool, # verbose
                c_void_p, # wksp
                c_void_p, #params
                # Diagnostic outputs
                c_ulong_p, # itCount
                array_1d_double, # resid
                array_1d_ulong, # rType
                c_double_p, # advanceTime
                c_double_p, # nextIterTime
                c_double_p # userTime
            ]
    else:
        __libvader.advanceCN.argtypes \
            = [ 
                # Time step and grid parameters
                c_double, #t
                c_double, #dt
                c_grid_p, #grid
                # Starting data
                array_1d_double, #col
                array_1d_double, #pres
                array_1d_double, #eInt
                array_1d_double, # mBnd
                array_1d_double, # eBnd
                array_1d_double, # mSrc
                array_1d_double, # eSrc
                # Equation of state parameters
                c_bool, # eos_func
                c_double, # gamma_val
                c_double, # delta_val
                # Dimensionless viscosity
                c_bool, # alpha_func
                c_double, # alpha_val
                # Inner boundary condition
                c_enum, # ibc_pres
                c_enum, # ibc_enth
                c_bool, # ibc_func
                c_double, # ibc_pres_val
                c_double, # ibc_enth_val
                # Outer boundary condition
                c_enum, # obc_pres
                c_enum, # obc_enth
                c_bool, # obc_func
                c_double, # obc_pres_val
                c_double, # obc_enth_val
                # Source terms
                c_bool, # massSrc_func
                c_double, # massSrc_val
                c_bool, # intEnSrc_func
                c_double, # intEnSrc_val
                # Control parameters
                c_double, # errTol
                c_double, # dtTol
                c_bool, # maxIter
                c_ulong, # interpOrder
                c_bool, # noUpdate
                c_bool, # verbose
                c_void_p, # wksp
                c_void_p, #params
                # Diagnostic outputs
                c_ulong # itCount
            ]
        __libvader.advanceBE.argtypes \
            = [ 
                # Time step and grid parameters
                c_double, #t
                c_double, #dt
                c_void_p, #grid
                # Starting data
                array_1d_double, #col
                array_1d_double, #pres
                array_1d_double, #eInt
                array_1d_double, # mBnd
                array_1d_double, # eBnd
                array_1d_double, # mSrc
                array_1d_double, # eSrc
                # Equation of state parameters
                c_bool, # eos_func
                c_double, # gamma_val
                c_double, # delta_val
                # Dimensionless viscosity
                c_bool, # alpha_func
                c_double, # alpha_val
                # Inner boundary condition
                c_enum, # ibc_pres
                c_enum, # ibc_enth
                c_bool, # ibc_func
                c_double, # ibc_pres_val
                c_double, # ibc_enth_val
                # Outer boundary condition
                c_enum, # obc_pres
                c_enum, # obc_enth
                c_bool, # obc_func
                c_double, # obc_pres_val
                c_double, # obc_enth_val
                # Source terms
                c_bool, # massSrc_func
                c_double, # massSrc_val
                c_bool, # intEnSrc_func
                c_double, # intEnSrc_val
                # Control parameters
                c_double, # errTol
                c_double, # dtTol
                c_bool, # maxIter
                c_ulong, # interpOrder
                c_bool, # noUpdate
                c_bool, # verbose
                c_void_p, # wksp
                c_void_p, #params
                # Diagnostic outputs
                c_ulong # itCount
            ]

    # Interface to main c vader routine
    __libvader.vader.restype = c_double
    if __testing_mode:
        __libvader.vader.argtypes \
            = [
                # Restart checkpoint
                c_char_p,   # restart file
                # Time parameters
                c_double,   # tStart
                c_double,   # tEnd
                # EOS parameters
                c_bool,     # eos_func
                c_double,   # gamma_val
                c_double,   # delta_val,
                # Viscosity parameters
                c_bool,     # alpha_func
                c_double,   # alpha_val
                # Inner boundary condition
                c_enum,     # ibc_pres
                c_enum,     # ibc_enth
                c_bool,     # ibc_func
                c_double,   # ibc_pres_val
                c_double,   # ibc_enth_val
                # Outer boundary condition
                c_enum,     # obc_pres
                c_enum,     # obc_enth
                c_bool,     # obc_func
                c_double,   # obc_pres_val
                c_double,   # obc_enth_val
                # Source function
                c_bool,     # massSrc_func
                c_double,   # massSrc_val
                c_bool,     # intEnSrc_func
                c_double,   # intEnSrc_val
                # Control and method parameters
                c_double,   # dtStart
                c_double,   # dtMin
                c_double,   # dtTol
                c_double,   # errTol
                c_double,   # maxDtIncrease
                c_ulong,    # maxIter
                c_ulong,    # interpOrder
                c_ulong,    # maxStep
                c_bool,     # useBE
                c_bool,     # preTimestep_func
                c_bool,     # postTimestep_func
                c_ulong,    # verbosity
                c_ulong,    # nSave
                c_double_p, # tSave
                c_ulong,    # nUserOut
                c_bool_p,   # userOutCum
                c_bool_p,   # writeCheckpoint
                c_char_p,   # checkname
                c_bool,     # userWriteCheckpoint
                c_bool,     # userReadCheckpoint
                # Grid
                c_grid_pp,   # grid
                # Initial conditions
                c_double_pp, # col_init
                c_double_pp, # pres_init
                c_double_pp, # eInt_init
                # User-defined parameters
                c_void_p,    # params
                # Diagnostic outputs
                c_ulong_p,   # nStep
                c_ulong_p,   # nIter
                c_ulong_p,   # nFail
                # Output values
                c_ulong_p,   # nOut
                c_double_pp, # tOut
                c_double_pp, # colOut
                c_double_pp, # presOut
                c_double_pp, # eIntOut
                c_double_pp, # mBndOut
                c_double_pp, # eBndOut
                c_double_pp, # mSrcOut
                c_double_pp, # eSrcOut
                c_double_pp, # userOut
                # Testing mode outputs
                array_1d_double,  # residSum
                array_1d_ulong,   # iterStep
                c_double_p,  # driverTime
                c_double_p,  # advanceTime
                c_double_p,  # nextIterTime
                c_double_p   # userTime
            ]
    else:
                __libvader.vader.argtypes \
            = [
                # Restart checkpoint
                c_char_p,   # restart file
                # Time parameters
                c_double,   # tStart
                c_double,   # tEnd
                # EOS parameters
                c_bool,     # eos_func
                c_double,   # gamma_val
                c_double,   # delta_val,
                # Viscosity parameters
                c_bool,     # alpha_func
                c_double,   # alpha_val
                # Inner boundary condition
                c_enum,     # ibc_pres
                c_enum,     # ibc_enth
                c_bool,     # ibc_func
                c_double,   # ibc_pres_val
                c_double,   # ibc_enth_val
                # Outer boundary condition
                c_enum,     # obc_pres
                c_enum,     # obc_enth
                c_bool,     # obc_func
                c_double,   # obc_pres_val
                c_double,   # obc_enth_val
                # Source function
                c_bool,     # massSrc_func
                c_double,   # massSrc_val
                c_bool,     # intEnSrc_func
                c_double,   # intEnSrc_val
                # Control and method parameters
                c_double,   # dtStart
                c_double,   # dtMin
                c_double,   # dtTol
                c_double,   # errTol
                c_double,   # maxDtIncrease
                c_ulong,    # maxIter
                c_ulong,    # interpOrder
                c_ulong,    # maxStep
                c_bool,     # useBE
                c_bool,     # preTimestep_func
                c_bool,     # postTimestep_func
                c_ulong,    # verbosity
                c_ulong,    # nSave
                c_double_p, # tSave
                c_ulong,    # nUserOut
                c_bool_p,   # userOutCum
                c_bool_p,   # writeCheckpoint
                c_char_p,   # checkname
                c_bool,     # userWriteCheckpoint
                c_bool,     # userReadCheckpoint
                # Grid
                c_grid_pp,   # grid
                # Initial conditions
                c_double_pp, # col_init
                c_double_pp, # pres_init
                c_double_pp, # eInt_init
                # User-defined parameters
                c_void_p,    # params
                # Diagnostic outputs
                c_ulong_p,   # nStep
                c_ulong_p,   # nIter
                c_ulong_p,   # nFail
                # Output values
                c_ulong_p,   # nOut
                c_double_pp, # tOut
                c_double_pp, # colOut
                c_double_pp, # presOut
                c_double_pp, # eIntOut
                c_double_pp, # mBndOut
                c_double_pp, # eBndOut
                c_double_pp, # mSrcOut
                c_double_pp, # eSrcOut
                c_double_pp  # userOut
            ]

##################################################################
# Interfaces to the allocation /setup routines                   #
##################################################################

def gridInit(grd):
    """
    Call the c grid intiailization routine, copying from a python grid
    object.

    Parameters
    ----------
    grd : class grid
       a python grid object

    Returns
    -------
    c_grd : c_void_p
       a pointer to a c grid object
    """

    loadlib()
    if grd.linear:
        linear_c = 1
    else:
        linear_c = 0
    return __libvader. \
        gridInit(grd.nr, grd.r_g, grd.r_h, 
                 grd.vphi_g, grd.vphi_h,
                 grd.beta_g, grd.beta_h, 
                 grd.psiEff_g, grd.psiEff_h,
                 grd.g_h, linear_c)


def gridInitKeplerian(nr, rmin, rmax, m, linear):
    """
    Call the c grid intiailization routine for a Keplerian rotation
    curve.

    Parameters
    ----------
    nr : int
       number of grid cells
    rmin : float
       radius of inner edge of cell 0
    rmax : float
       radius of outer edge of cell nr-1
    m : float
       mass of central object, in CGS units
    linear : bool
       if True, the grid is uniformly spaced in r; otherwise it is
       uniformly spaced in ln r

    Returns
    -------
    c_grd : c_void_p
       a pointer to a c grid object
    """
    loadlib()
    if linear:
        linear_c = 1
    else:
        linear_c = 0
    return __libvader. \
        gridInitKeplerian(nr, rmin, rmax, m, linear_c)


def gridInitFlat(nr, rmin, rmax, vphi, linear):
    """
    Call the c grid intiailization routine for a flat rotation
    curve.

    Parameters
    ----------
    nr : int
       number of grid cells
    rmin : float
       radius of inner edge of cell 0
    rmax : float
       radius of outer edge of cell nr-1
    vphi : float
       rotation curve velocity
    linear : bool
       if True, the grid is uniformly spaced in r; otherwise it is
       uniformly spaced in ln r

    Returns
    -------
    c_grd : c_void_p
       a pointer to a c grid object
    """
    loadlib()
    if linear:
        linear_c = 1
    else:
        linear_c = 0
    return __libvader. \
        gridInitKeplerian(nr, rmin, rmax, vphi, linear_c)


def wkspAlloc(nr):
    """
    Call the c workspace allocation routine.

    Parameters
    ----------
    nr : int
       number of grid cells

    Returns
    -------
    c_wksp : c_void_p
       a pointer to a c workspace object
    """
    loadlib()
    return __libvader.wkspAlloc(nr)


def gridFree(cgrd):
    """
    Call the c grid de-allocation routine.

    Parameters
    ----------
    cgrd : c_void_p
       the object to be de-allocated

    Returns
    -------
    Nothing
    """
    loadlib()
    __libvader.gridFree(cgrd)


def wkspFree(cwksp):
    """
    Call the c workspace de-allocation routine.

    Parameters
    ----------
    cwksp : c_void_p
       the object to be de-allocated

    Returns
    -------
    Nothing
    """
    loadlib()
    __libvader.wkspFree(cwksp)


##################################################################
# Interface to the b-spline rotation curve fitting routine       #
##################################################################

def rotCurveSpline(rTab, vTab, r, bsplineDegree=6,
                   bsplineBreakpoints=15):
    """
    Calls the c rotation curve b-spline fitting routine to compute a
    fit to a tabulated rotation curve.

    Parameters
    ----------
    rTab : array
       array of radii in rotation curve table; must be in ascending
       order
    vTab: array
       array of velocities in rotation curve table
    r : array
       array of positions at which the rotation curve is to be fit;
       minimum value must be >= rTab[0], and maximum value must be <=
       rTab[-1]
    bsplineDegree : int
       order of basis spline fit
    bsplineBreakpoints : int
       number of breakpoints to use

    Returns
    -------
    vphi : array
       fitted values of vphi at fit points
    psi : array
       values of gravitational potential psi at fit points
    beta : array
       values of dln vphi / dln r at fit points
    """

    loadlib()

    # Safety check
    if bsplineBreakpoints + bsplineDegree - 2 > len(rTab):
        raise ValueError(
            ("rotCurveSpline invoked with {:d} data points "
             +"for {:d} coefficients; ndata must be >= ncoef "
             +"= nbreakpoints + degree - 2").
            format(len(rTab), 
                   bsplineBreakpoints+bsplineDegree-2))

    # Call function
    vphi = np.zeros(len(r))
    beta = np.zeros(len(r))
    psi = np.zeros(len(r))
    __libvader.rotCurveSpline(rTab, vTab, len(rTab),
                              bsplineDegree, bsplineBreakpoints,
                              r, len(r), vphi, psi, beta)
    return vphi, psi, beta


##################################################################
# Routine to parse a parameter dictionary and produce another    #
# dict containig the values required to call the c routines      #
##################################################################

def cDict(paramDict, **kwargs):
    """
    Take a paramDict and produce a dict containing all the values
    required to call the c vader routines. This involves converting
    variable types and setting defaults for parameters not in the
    paramDict.

    Parameters
    ----------
    paramDict : dict
       the python dict to start from
    kwargs : dict
       any additional keywords set are added to the paramDict before it
       is parsed to produce the c dict. Keys already in paramDict are
       overridden, but paramDict itself is not altered.

    Returns
    -------
    cD : dict
       a dict of values suitable for calling the c routines.
    """

    # Create a copy of the input dict that we can edit
    pDict = deepcopy(paramDict)

    # Apply keyword overrides
    for k in kwargs:
        pDict[k] = kwargs[k]

    # Check that mandatory parameters are present
    mandatory = ['alpha', 'gamma', 'ibc_pres_type', 'ibc_enth_type',
                 'ibc_pres_val', 'obc_pres_type', 'obc_enth_type',
                 'obc_pres_val']
    for m in mandatory:
        if not m in pDict:
            raise ValueError("Error: paramDict must contain key "+m)

    # For parameters that are not mandatory, apply defaults
    for k in __defDict.keys():
        if k not in pDict:
            pDict[k] = __defDict[k]

    # Prepare the c dict
    cD = {}

    # Time parameters
    cD['dt_start'] = float(pDict['dt_start'])

    # EOS parameters
    if pDict['gamma'] == 'c_func':
        cD['eos_func'] = True
        cD['gamma_val'] = 0.0
        cD['delta_val'] = 0.0
    else:
        cD['eos_func'] = False
        cD['gamma_val'] = float(pDict['gamma'])
    if pDict['delta'] == 'c_func':
        cD['eos_func'] = True
        cD['gamma_val'] = 0.0
        cD['delta_val'] = 0.0
    else:
        cD['delta_val'] = float(pDict['delta'])

    # Viscosity parameters
    if pDict['alpha'] == 'c_func':
        cD['alpha_func'] = True
        cD['alpha_val'] = 0.0
    else:
        cD['alpha_func'] = False
        cD['alpha_val'] = float(pDict['alpha'])

    # Inner boundary condition parameters
    if pDict['ibc_pres_type'] == 'fixed_mass_flux':
        cD['ibc_pres'] = 0
    elif pDict['ibc_pres_type'] == 'fixed_torque_flux':
        cD['ibc_pres'] = 1
    elif pDict['ibc_pres_type'] == 'fixed_torque':
        cD['ibc_pres'] = 2
    else:
        raise ValueError("Error: unknown ibc_pres_type " + \
              str(pDict['ibc_pres_type']))
    if pDict['ibc_pres_val'] == 'c_func':
        cD['ibc_func'] = True
        cD['ibc_pres_val'] = 0.0
        cD['ibc_enth_val'] = 0.0
    else:
        cD['ibc_func'] = False
        cD['ibc_pres_val'] = float(pDict['ibc_pres_val'])
    if pDict['ibc_enth_type'] == 'fixed_value':
        cD['ibc_enth'] = 0
    elif pDict['ibc_enth_type'] == 'fixed_gradient':
        cD['ibc_enth'] = 1
    else:
        raise ValueError("Error: unknown ibc_enth_type " + \
              str(pDict['ibc_enth_type']))
    if not cD['ibc_func']:
        if pDict['ibc_enth_val'] == 'c_func':
            cD['ibc_func'] = 1
            cD['ibc_pres_val'] = 0.0
            cD['ibc_enth_val'] = 0.0
        else:
            cD['ibc_enth_val'] = float(pDict['ibc_enth_val'])

    # Outer boundary condition parameters
    if pDict['obc_pres_type'] == 'fixed_mass_flux':
        cD['obc_pres'] = 0
    elif pDict['obc_pres_type'] == 'fixed_torque_flux':
        cD['obc_pres'] = 1
    elif pDict['obc_pres_type'] == 'fixed_torque':
        cD['obc_pres'] = 2
    else:
        raise ValueError("Error: unknown obc_pres_type " + \
              str(pDict['obc_pres_type']))
    if pDict['obc_pres_val'] == 'c_func':
        cD['obc_func'] = True
        cD['obc_pres_val'] = 0.0
        cD['obc_enth_val'] = 0.0
    else:
        cD['obc_func'] = False
        cD['obc_pres_val'] = float(pDict['obc_pres_val'])
    if pDict['obc_enth_type'] == 'fixed_value':
        cD['obc_enth'] = 0
    elif pDict['obc_enth_type'] == 'fixed_gradient':
        cD['obc_enth'] = 1
    else:
        raise ValueError("Error: unknown obc_enth_type " + \
              str(pDict['obc_enth_type']))
    if not cD['obc_func']:
        if pDict['obc_enth_val'] == 'c_func':
            cD['obc_func'] = True
            cD['obc_pres_val'] = 0.0
            cD['obc_enth_val'] = 0.0
        else:
            cD['obc_enth_val'] = float(pDict['obc_enth_val'])

    # Source function parameters
    if pDict['mass_src'] == 'c_func':
        cD['mass_src_func'] = True
        cD['mass_src_val'] = 0.0
    else:
        cD['mass_src_func'] = False
        cD['mass_src_val'] = float(pDict['mass_src'])
    if pDict['int_en_src'] == 'c_func':
        cD['int_en_src_func'] = True
        cD['int_en_src_val'] = 0.0
    else:
        cD['int_en_src_func'] = False
        cD['int_en_src_val'] = float(pDict['int_en_src'])

    # Control parameters
    cD['err_tol'] = float(pDict['err_tol'])
    cD['dt_tol'] = float(pDict['dt_tol'])
    cD['max_iter'] = int(pDict['max_iter'])
    cD['interp_order'] = int(pDict['interp_order'])
    if (cD['interp_order'] < 1) or (cD['interp_order'] > 3):
        raise ValueError("interp_order must be 1, 2, or 3")
    cD['max_dt_increase'] = float(pDict['max_dt_increase'])
    cD['dt_min'] = float(pDict['dt_min'])
    cD['max_step'] = int(pDict['max_step'])
    if pDict['method'] == 'CN':
        cD['use_BE'] = False
    elif pDict['method'] == 'BE':
        cD['use_BE'] = True
    else:
        raise ValueError("method must be 'CN' or 'BE'")
    if pDict['pre_timestep'] == 'c_func':
        cD['pre_timestep_func'] = True
    else:
        cD['pre_timestep_func'] = False
    if pDict['post_timestep'] == 'c_func':
        cD['post_timestep_func'] = True
    else:
        cD['post_timestep_func'] = False
    if pDict['user_check_read'] == 'c_func':
        cD['check_read'] = True
    else:
        cD['check_read'] = False
    if pDict['user_check_write'] == 'c_func':
        cD['check_write'] = True
    else:
        cD['check_write'] = False
    cD['verbosity'] = int(pDict['verbosity'])

    # Return c dict
    return cD


##################################################################
# Interface to main vader routine                                #
##################################################################

def vader(tStart, tEnd, paramDict, col=None, pres=None, eInt=None,
          grd=None, restart=None, nSave=50, saveTimes=None,
          c_params=None, nUserOut=0, userCum=None, checkpoint=None,
          checkname=None, **kwargs):
    """
    Parameters
    ----------
    tStart : float
       simulation start time
    tEnd : float
       simulation end time
    paramDict : dict
       dict of simulation parameters
    col : arraylike
       array of initial column densities; on return, this will contain
       the final column density; only required if restart is None
    pres : arraylike
       array of initial pressures; on return, this will contain the
       final pressures; only required if restart is None
    eInt : arraylike
       array of starting internal energies; on return, this will
       contain the final internal energies; only required if gamma or
       delta are specified as c_func instead of constants, and if
       restart is None
    grd : grid
       grid object describing the computional grid; only needed if
       restart is None
    restart : string
       a string giving the name of a checkpoint from which to restart
    nSave : int
       number of times at which to save the state of the simulation;
       times are uniformly spaced from tStart to tEnd, inclusive; set
       to 0 just to return the final state
    saveTimes : array
       exact times at which to save the state of the simulation; if
       set, this overrides nSave
    c_params : c_void_p
       pointer to parameters to be passed into the c functions
    nUserOut : int
       number of user-defined output fields that the c code will
       output
    userCum : None | bool | array of bool, shape (nUserOut)
       if True, an output variable is computed cumulatively, meaning
       that it is summed over all time steps; default is
       non-cumulative; user can provide a single value for all user
       output variables, or an array giving one value per variable
    checkpoint : None | bool | arraylike of bool, shape (nSave)
       if True, each saved output is also written out as a checkpoint
       in a file named checkname_NNNN.vader, where NNNN is the output
       number; if this is an array, only outputs for which
       checkpoint[i] is True are written as output; default is no
       checkpointing
    checkname : None | string
       base name of checkpoint file; it is an error if this is None
       and checkpoint is not either None or False
    kwargs : dict
       any additional keywords specified here are appended to the
       paramDict before the simulation is run

    Returns
    -------
    out : nametuple

       a namedtuple containing the following fields:

       grid : grid
          the computational grid used in the simulation
       tFin : float
          time at which simulation ended
       nStep : long
          total number of simulation timesteps taken
       nIter : long
          total number of implicit iterations, summed over all time
          steps
       nFail : long
          total number of times the implicit solver failed to converge

       The following items are all None if nOut < 1:

       tOut : array, shape (nOut)
          times at which output is stored
       colOut : array, shape (nOut,nr)
          2D array of column densities stored at specified times
       presOut : array, shape (nOut,nr)
          2D array of pressures stored at specified times
       eIntOut : array, shape (nOut,nr)
          2D array of internal energies stored at specified times;
          returned only if gamma or delta are set to c_func;
          otherwise this is None
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
          mass_src is set to c_func; otherwise this is None
       eSrcOut : array, shape (nOut,nr)
          cumulative energy per unit area added to each cell by the
          source terms up to the specified time; returned only if
          mass_src or int_en_src is set to c_func; otherwise this is
          None
       userOut : array, shape (nOut, nUserOut, nr)
          user-defined outputs, written by the user-defined c code;
          returned only if nUserOut > 0; otherwise this is None

       The following items are included only if the code was compiled
       with -DTESTING_MODE:

       residSum : array, shape (max_iter)
          sum of residuals after a given iteration number, summed over
          all time steps
       iterStep : array of ulong, shape (max_step)
          total number of iterations performed in each time step
       driverTime : c_double
          total time in seconds spent in driver routine
       advanceTime : c_double
          total time in seconds spent in advance routine
       nextIterTime : c_double
          total time in seconds spent in getNextIterate routine
       userTime : c_double
          total time in seconds spent in user routines
    """

    loadlib()

    # Get the paramDict ready for passing to c
    cD = cDict(paramDict, **kwargs)

    # Safety checks
    if restart is None:
        if (grd is None):
            raise ValueError("must set either restart or grd!")
        if (col is None) or (pres is None):
            raise ValueError("must set either restart or initial "+
                             "col and pres!")
        if cD['eos_func'] and (eInt is None):
            raise ValueError("must set eInt if eos_func is True")
        elif not cD['eos_func'] and (eInt is not None):
            warn("warning: eInt will not be evolved for EOS with"
                 + " constant gamma, delta")
        
    # If this is not a restart, make sure that we have a c version of
    # the computational grid
    if restart is None and grd.c_grd is None:
        grd.c_grd = gridInit(grd)
        grd_pp = pointer(grd.c_grd)
    else:
        grd_pp = pointer(c_grid_p())

    # Prepare holders for diagnostic output
    nStep = c_ulong(0)
    nIter = c_ulong(0)
    nFail = c_ulong(0)
    if __testing_mode:
        driverTime = c_double(0.0)
        advanceTime = c_double(0.0)
        nextIterTime = c_double(0.0)
        userTime = c_double(0.0)
        residSum = np.zeros(cD['max_iter'])
        iterStep = np.zeros(cD['max_step'], dtype=c_ulong)

    # Set times at which to save
    if saveTimes is None:
        if nSave > 1:
            tSave = tStart + \
                   (tEnd - tStart)*np.arange(nSave)/(nSave-1)
            tSave_p = tSave.ctypes.data_as(c_double_p)
        elif nSave == 1:
            tSave_p = tSave.ctypes.data_as(c_double_p)
            tSave = np.zeros(1) + tStart + (tEnd-tStart)/2.0
        else:
            tSave = None
            tSave_p = c_double_p()
    else:
        tSave = np.array(saveTimes, dtype=c_double)
        tSave_p = tSave.ctypes.data_as(c_double_p)
        nSave = len(tSave)

    # If we have user outputs, specify which of them are cumulative
    if nUserOut > 0:
        userCum_c = np.zeros(nUserOut, dtype=c_bool)
        if userCum is not None:
            userCum_c[:] = userCum
        userCum_p = userCum_c.ctypes.data_as(c_bool_p)
    else:
        userCum_p = c_bool_p()

    # Set up checkpointing
    if checkpoint is None or checkpoint is False:
        checkpoint_p = c_bool_p()
        checkname_p = c_char_p()
    elif checkpoint is True:
        chk = np.array([True]*nSave, dtype=c_bool)
        checkpoint_p = chk.ctypes.data_as(c_bool_p)
        if checkname is None:
            raise ValueError("checkname cannot be None " +
                             "if checkpoint is True")
        checkname_p = checkname
    else:
        chk = np.array(checkpoint, dtype=c_bool)
        checkpoint_p = chk.ctypes.data_as(c_bool_p)
        if checkname is None:
            raise ValueError("checkname cannot be None " +
                             "if checkpoint is True")
        checkname_p = checkname

    # Set pointers to the column density, pressure, and internal
    # energy
    if restart is None:
        col = np.array(col, dtype=c_double)
        col_pp = pointer(col.ctypes.data_as(c_double_p))
        pres = np.array(pres, dtype=c_double)
        pres_pp = pointer(pres.ctypes.data_as(c_double_p))
        if cD['eos_func']:
            eInt = np.array(eInt, dtype=c_double)
            eInt_pp = pointer(eInt.ctypes.data_as(c_double_p))
        else:
            eInt_pp = c_double_pp()
    else:
        col_pp = pointer(c_double_p())
        pres_pp = pointer(c_double_p())
        if cD['eos_func']:
            eInt_pp = pointer(c_double_p())
        else:
            eInt_pp = c_double_pp()

    # Set up pointers to hold outputs; these are pointers to pointers
    # to c double, and on return the pointers to c double will contain
    # allocated arrays that we can use
    nOut = c_ulong(0)
    tOut_pp = pointer(c_double_p())
    colOut_pp = pointer(c_double_p())
    presOut_pp = pointer(c_double_p())
    eIntOut_pp = pointer(c_double_p())
    mBndOut_pp = pointer(c_double_p())
    eBndOut_pp = pointer(c_double_p())
    mSrcOut_pp = pointer(c_double_p())
    eSrcOut_pp = pointer(c_double_p())
    userOut_pp = pointer(c_double_p())

    # Call the main c vader function to run the simulation
    if __testing_mode:
        tFin = __libvader.vader(
            # Restart file
            restart,
            # Time parameters
            tStart, tEnd,
            # Equation of state parameters
            cD['eos_func'], cD['gamma_val'], cD['delta_val'],
            # Viscosity parameters
            cD['alpha_func'], cD['alpha_val'],
            # Inner boundary condition parameters
            cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
            cD['ibc_pres_val'], cD['ibc_enth_val'],
            # Outer boundary condition parameters
            cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
            cD['obc_pres_val'], cD['obc_enth_val'],
            # Source function parameters
            cD['mass_src_func'], cD['mass_src_val'],
            cD['int_en_src_func'], cD['int_en_src_val'],
            # Control and method parameters
            cD['dt_start'], cD['dt_min'], cD['dt_tol'],
            cD['err_tol'], cD['max_dt_increase'],
            cD['max_iter'], cD['interp_order'],
            cD['max_step'], cD['use_BE'],
            cD['pre_timestep_func'], cD['post_timestep_func'],
            cD['verbosity'],
            # Output control parameters
            nSave, tSave_p, nUserOut, userCum_p,
            checkpoint_p, checkname_p,
            cD['check_read'], cD['check_write'],
            # Grid
            grd_pp,
            # Initial conditions
            col_pp, pres_pp, eInt_pp,
            # Parameters
            c_params,
            # Diagnostic outputs
            nStep, nIter, nFail,
            # Outputs
            nOut, tOut_pp, colOut_pp, presOut_pp,
            eIntOut_pp, mBndOut_pp, eBndOut_pp,
            mSrcOut_pp, eSrcOut_pp, userOut_pp,
            # Testing mode parameters
            residSum, iterStep, driverTime, advanceTime,
            nextIterTime, userTime)
    else:
        tFin = __libvader.vader(
            # Restart file
            restart,
            # Time parameters
            tStart, tEnd,
            # Equation of state parameters
            cD['eos_func'], cD['gamma_val'], cD['delta_val'],
            # Viscosity parameters
            cD['alpha_func'], cD['alpha_val'],
            # Inner boundary condition parameters
            cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
            cD['ibc_pres_val'], cD['ibc_enth_val'],
            # Outer boundary condition parameters
            cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
            cD['obc_pres_val'], cD['obc_enth_val'],
            # Source function parameters
            cD['mass_src_func'], cD['mass_src_val'],
            cD['int_en_src_func'], cD['int_en_src_val'],
            # Control and method parameters
            cD['dt_start'], cD['dt_min'], cD['dt_tol'],
            cD['err_tol'], cD['max_dt_increase'],
            cD['max_iter'], cD['interp_order'],
            cD['max_step'], cD['use_BE'],
            cD['pre_timestep_func'], cD['post_timestep_func'],
            cD['verbosity'],
            # Output control parameters
            nSave, tSave_p, nUserOut, userCum_p,
            checkpoint_p, checkname_p,
            cD['check_read'], cD['check_write'],
            # Grid
            grd_pp,
            # Initial conditions
            col_pp, pres_pp, eInt_pp,
            # Parameters
            c_params,
            # Diagnostic outputs
            byref(nStep), byref(nIter), byref(nFail),
            # Outputs
            nOut, tOut_pp, colOut_pp, presOut_pp,
            eIntOut_pp, mBndOut_pp, eBndOut_pp,
            mSrcOut_pp, eSrcOut_pp, userOut_pp)

    # If needed, construct a grid from the buffer that we get back,
    # and construct the final column density, pressure, and internal
    # energy. Free the buffers for these when we're done.
    if restart:
        grd = grid.grid(paramDict, c_grd = grd_pp[0][0])
        col = np.array(col_pp[0][:grd.nr])
        pres = np.array(pres_pp[0][:grd.nr])
        if cD['eos_func']:
            eInt = np.array(eInt_pp[0][:grd.nr])
        null_ptr = c_double_pp()
        __libvader.outputFree(null_ptr, col_pp, pres_pp,
                              eInt_pp, null_ptr, null_ptr,
                              null_ptr, null_ptr, null_ptr)
        
    # Convert the outputs to arrays; note that we construct the arrays
    # in a way that guarantees we actually copy the data out of the
    # buffers returned by c, and then we de-allocate them. This is
    # necessary to avoid memory leaks. The buffers returned by the c
    # code were allocated in c, and will not be garbage-collected when
    # the numpy arrays pass out of scope.
    if nOut.value > 0:
        tOut = np.array(tOut_pp[0][:nOut.value])
        colOut = np.array(colOut_pp[0][:nOut.value*grd.nr]).\
            reshape((nOut.value, grd.nr))
        presOut = np.array(presOut_pp[0][:nOut.value*grd.nr]).\
            reshape((nOut.value, grd.nr))
        if cD['eos_func']:
            eIntOut = np.array(eIntOut_pp[0][:nOut.value*grd.nr]).\
                      reshape((nOut.value, grd.nr))
        else:
            eIntOut = None
        mBndOut = np.array(mBndOut_pp[0][:nOut.value*2]).\
            reshape((nOut.value, 2))
        eBndOut = np.array(eBndOut_pp[0][:nOut.value*2]).\
            reshape((nOut.value, 2))
        if cD['mass_src_func']:
            mSrcOut = np.array(mSrcOut_pp[0][:nOut.value*grd.nr]).\
                      reshape((nOut.value, grd.nr))
        else:
            mSrcOut = None
        if cD['mass_src_func'] or cD['int_en_src_func']:
            eSrcOut = np.array(eSrcOut_pp[0][:nOut.value*grd.nr]).\
                      reshape((nOut.value, grd.nr))
        else:
            eSrcOut = None
        if nUserOut > 0:
            userOut = np.array(
                userOut_pp[0][:nOut.value*grd.nr*nUserOut]).\
                reshape((nOut.value, nUserOut, grd.nr))
        else:
            userOut = None

        # De-allocate the memory returned by c
        __libvader.outputFree(tOut_pp, colOut_pp, presOut_pp,
                              eIntOut_pp, mBndOut_pp, eBndOut_pp,
                              mSrcOut_pp, eSrcOut_pp, userOut_pp)
    else:

        # Set all outputs to None
        tout = None
        colOut = None
        presOut = None
        eIntOut = None
        mBndOut = None
        eBndOut = None
        mSrcOut = None
        eSrcOut = None
        userOut = None
        
    # Construct the object to return
    if __testing_mode:
        out_obj = namedtuple(
            'VADER_out',
            'grid tFin nStep nIter nFail '+
            'tOut colOut presOut eIntOut mBndOut eBndOut '+
            'mSrcOut eSrcOut userOut' +
            'residSum iterStep driverTime advanceTime '+
            'nextIterTime userTime')
        out = out_obj(grd, tFin, nStep.value, nIter.value, nFail.value,
                      tOut, colOut, presOut, eIntOut, mBndOut,
                      eBndOut, mSrcOut, eSrcOut, userOut,
                      residSum, iterStep, driverTime.value,
                      advanceTime.value, nextIterTime.value,
                      userTime.value)
    else:
        out_obj = namedtuple(
            'VADER_out',
            'grid tFin nStep nIter nFail '+
            'tOut colOut presOut eIntOut mBndOut eBndOut '+
            'mSrcOut eSrcOut userOut')
        out = out_obj(grd, tFin, nStep.value, nIter.value, nFail.value,
                      tOut, colOut, presOut, eIntOut, mBndOut,
                      eBndOut, mSrcOut, eSrcOut, userOut)

    # Return
    return out
       
##################################################################
# Interface to c driver routine                                  #
##################################################################

def driver(tStart, tEnd, grd, col, pres, paramDict, nOut=50,
           outTimes=None, eInt=None, c_params=None, nUserOut=0,
           userCum=None, checkpoint=None, checkname=None,
           restart=None, **kwargs):
    """
    Main driver routine to run a vader simulation.

    Parameters
    ----------
    tStart : float
       simulation start time
    tEnd : float
       simulation end time
    grd : class grid
       grid object describing the simulation grid
    col : array
       array of starting column densities
    pres : array
       array of starting pressures
    paramDict : dict
       dict of simulation parameters
    nOut : int
       number of times at which to store output; outputs are only
       stored if nOut >= 1
    outTimes : array
       exact times at which to store output; if set, this overrides
       nOut
    eInt : array
       array of starting internal energies; must be set if gamma or
       delta are specified as c_func instead of constants, ignored
       otherwise
    c_params : c_void_p
       pointer to parameters to be passed into the c functions
    nUserOut : int
       number of user-defined output fields that the c code will
       output
    userCum : None | bool | array of bool, shape (nUserOut)
       if True, an output variable is computed cumulatively, meaning
       that it is summed over all time steps; default is
       non-cumulative; user can provide a single value for all user
       output variables, or an array giving one value per variable
    checkpoint : None | bool | arraylike of bool, shape (nOut)
       if True, each saved output is also written out as a checkpoint
       in a file named checkname_NNNN.vader, where NNNN is the output
       number; if this is an array, only outputs for which
       checkpoint[i] is True are written as output; default is no
       checkpointing
    checkname : None | string
       base name of checkpoint file; it is an error if this is None
       and checkpoint is not either None or False
    restart : string | file
       if not None, the calculation is restarted from a checkpoint;
       the value of restart can be either a string giving the name of
       the checkpoint file, of a file handle
    kwargs : dict
       any additional keywords specified here are appended to the
       paramDict before the simulation is run

    Returns
    -------
    out : nametuple
       out is a namedtuple containing the following output items:
       tFin : float
          time at which simulation ended
       The following items are all None if nOut < 1:
       tOut : array, shape (nOut)
          times at which output is stored
       colOut : array, shape (nOut,nr)
          2D array of column densities stored at specified times
       presOut : array, shape (nOut,nr)
          2D array of pressures stored at specified times
       eIntOut : array, shape (nOut,nr)
          2D array of internal energies stored at specified times;
          returned only if gamma or delta are set to c_func;
          otherwise this is None
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
          mass_src is set to c_func; otherwise this is None
       eSrcOut : array, shape (nOut,nr)
          cumulative energy per unit area added to each cell by the
          source terms up to the specified time; returned only if
          mass_src or int_en_src is set to c_func; otherwise this is
          None
       userOut : array, shape (nOut, nUserOut, nr)
          user-defined outputs, written by the user-defined c code;
          returned only if nUserOut > 0; otherwise this is None
       nStep : long
          total number of simulation timesteps taken
       nIter : long
          total number of implicit iterations, summed over all time
          steps
       nFail : long
          total number of times the implicit solver failed to converge
       residSum : array, shape (max_iter)
          sum of residuals after a given iteration number, summed over
          all time steps; returned only if the code is compiled with
          -DTESTING_MODE
       iterStep : array of uintc, shape (max_step)
          total number of iterations performed in each time step;
          returned only if the code is compiled with -DTESTING_MODE
       driverTime : c_double
          total time in seconds spent in driver routine; returned
          only if the code is compiled with -DTESTING_MODE
       advanceTime : c_double
          total time in seconds spent in advance routine; returned
          only if the code is compiled with -DTESTING_MODE
       nextIterTime : c_double
          total time in seconds spent in getNextIterate routine; returned
          only if the code is compiled with -DTESTING_MODE
       userTime : c_double
          total time in seconds spent in user routines; returned
          only if the code is compiled with -DTESTING_MODE
    """

    loadlib()

    # Get the paramDict ready for passing to c
    cD = cDict(paramDict, **kwargs)

    # Safety check: make sure that we have an eInt value if we're
    # using a complex EOS, and warn if we were given one but we have a
    # simple EOS
    if cD['eos_func'] == 1 and eInt is None:
        raise ValueError("must set eInt if eos_func == 1")
    elif cD['eos_func'] == 0 and eInt is not None:
        print("warning: eInt will not be evolved for EOS with"
              + " constant gamma, delta")

    # eInt must be a numpy array, so create a dummy one if necessary
    if eInt is None:
        eInt = np.zeros(1)

    # Prepare storage for output
    nStep = c_ulong(0)
    nIter = c_ulong(0)
    nFail = c_ulong(0)
    if __testing_mode:
        driverTime = c_double(0.0)
        advanceTime = c_double(0.0)
        nextIterTime = c_double(0.0)
        userTime = c_double(0.0)
    if outTimes is None:
        if nOut > 1:
            tOut = tStart + \
                   (tEnd - tStart)*np.arange(nOut)/(nOut-1)
        elif nOut == 1:
            tOut = np.zeros(1) + tStart + (tEnd-tStart)/2.0
        else:
            tOut = np.zeros(1)
    else:
        tOut = np.copy(outTimes)
        nOut = len(tOut)
    if nOut > 0:
        colOut = np.zeros((nOut, grd.nr))
        presOut = np.zeros((nOut, grd.nr))
        if cD['eos_func'] == 1:
            eIntOut = np.zeros((nOut, grd.nr))
        else:
            eIntOut = np.zeros((1,1))
        mBndOut = np.zeros((nOut, 2))
        eBndOut = np.zeros((nOut, 2))
        if cD['mass_src_func'] == 1:
            mSrcOut = np.zeros((nOut, grd.nr))
            eSrcOut = np.zeros((nOut, grd.nr))
        elif cD['int_en_src_func'] == 1:
            mSrcOut = np.zeros((1, grd.nr))
            eSrcOut = np.zeros((nOut, grd.nr))
        else:
            mSrcOut = np.zeros((1, grd.nr))
            eSrcOut = np.zeros((1, grd.nr))
        if nUserOut > 0:
            userOut = np.zeros((nOut, nUserOut, grd.nr))
        else:
            userOut = np.zeros((1,1,1))
    else:
        colOut = np.zeros((1, grd.nr))
        presOut = np.zeros((1, grd.nr))
        if cD['eos_func'] == 1:
            eIntOut = np.zeros((1, grd.nr))
        else:
            eIntOut = np.zeros((1,1))
        mBndOut = np.zeros((1, 2))
        eBndOut = np.zeros((1, 2))
        if cD['mass_src_func'] == 1:
            mSrcOut = np.zeros((1, grd.nr))
            eSrcOut = np.zeros((1, grd.nr))
        elif cD['int_en_src_func'] == 1:
            mSrcOut = np.zeros((1, grd.nr))
            eSrcOut = np.zeros((1, grd.nr))
        else:
            mSrcOut = np.zeros((1, grd.nr))
            eSrcOut = np.zeros((1, grd.nr))
        if nUserOut > 0:
            userOut = np.zeros((1, nUserOut, grd.nr))
        else:
            userOut = np.zeros((1,1,1))
    if nUserOut > 0:
        userCum_c = np.zeros(nUserOut, dtype=c_bool)
        if userCum is not None:
            userCum_c[:] = userCum
    else:
        userCum_c = np.zeros(1, dtype=c_bool)
    if __testing_mode:
        residSum = np.zeros(cD['max_iter'])
        iterStep = np.zeros(cD['max_step'], dtype=np.uintc)

    # Set up checkpointing
    if checkpoint is None:
        c_checkpoint = None
        c_checkname = None
    elif checkpoint is False:
        c_checkpoint = None
        c_checkname = None
    elif checkpoint is True:
        chk = np.array([1]*nOut, dtype=c_uint)
        c_checkpoint = chk.ctypes.data_as(ctypes.POINTER(c_uint))
        if checkname is None:
            raise ValueError("checkname cannot be None " +
                             "if checkpoint is True")
        c_checkname = str(checkname)
    else:
        chk = np.array(checkpoint, dtype=c_uint)
        c_checkpoint = chk.ctypes.data_as(ctypes.POINTER(c_uint))
        if checkname is None:
            raise ValueError("checkname cannot be None " +
                             "if checkpoint is True")
        c_checkname = str(checkname)

    # Call c library
    if __testing_mode:
        tFin = __libvader. \
               driver(tStart, tEnd, cD['dt_start'], grd.c_grd,
                      col, pres, eInt,
                      cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                      cD['alpha_func'], cD['alpha_val'],
                      cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                      cD['ibc_pres_val'], cD['ibc_enth_val'],
                      cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                      cD['obc_pres_val'], cD['obc_enth_val'],
                      cD['mass_src_func'], cD['mass_src_val'],
                      cD['int_en_src_func'], cD['int_en_src_val'],
                      cD['err_tol'], cD['dt_tol'],
                      cD['max_iter'], cD['interp_order'],
                      cD['max_dt_increase'], cD['dt_min'], cD['max_step'],
                      cD['use_BE'], cD['pre_timestep_func'],
                      cD['post_timestep_func'],
                      cD['verbosity'], grd.c_wksp,
                      c_params, nOut, tOut, colOut, presOut, eIntOut,
                      mBndOut, eBndOut, mSrcOut, eSrcOut,
                      nUserOut, userOut, userCum_c,
                      c_checkpoint, c_chekname,
                      nStep, nIter, nFail, residSum, iterStep,
                      driverTime, advanceTime, nextIterTime, userTime)
    else:
        tFin = __libvader. \
               driver(tStart, tEnd, cD['dt_start'], grd.c_grd,
                      col, pres, eInt,
                      cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                      cD['alpha_func'], cD['alpha_val'],
                      cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                      cD['ibc_pres_val'], cD['ibc_enth_val'],
                      cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                      cD['obc_pres_val'], cD['obc_enth_val'],
                      cD['mass_src_func'], cD['mass_src_val'],
                      cD['int_en_src_func'], cD['int_en_src_val'],
                      cD['err_tol'], cD['dt_tol'],
                      cD['max_iter'], cD['interp_order'],
                      cD['max_dt_increase'], cD['dt_min'], cD['max_step'],
                      cD['use_BE'], cD['pre_timestep_func'],
                      cD['post_timestep_func'],
                      cD['verbosity'], grd.c_wksp,
                      c_params, nOut, tOut, colOut, presOut, eIntOut,
                      mBndOut, eBndOut, mSrcOut, eSrcOut,
                      nUserOut, userOut, userCum_c,
                      c_checkpoint, c_checkname,
                      nStep, nIter, nFail)

    # If we terminated early, trim the output appropriately
    if nOut > 0:
        if tFin < tOut[-1]:
            idx = tOut <= tFin
            tOut = tOut[idx]
            colOut = colOut[idx,:]
            presOut = presOut[idx,:]
            if cD['eos_func'] == 1:
                eIntOut = eIntOut[idx,:]
            mBndOut = mBndOut[idx,:]
            eBndOut = eBndOut[idx,:]
            if cD['mass_src_func'] == 1:
                mSrcOut = mSrcOut[idx,:]
                eSrcOut = eSrcOut[idx,:]
            elif cD['int_en_src_func'] == 1:
                eSrcOut = eSrcOut[idx,:]
            if nUserOut > 0:
                userOut = userOut[idx,:,:]
    else:
        tOut = None
        colOut = None
        presOut = None
        eIntOut = None
        mBndOut = None
        eBndOut = None
        mSrcOut = None
        eSrcOut = None

    # Construct the object to return, then return it
    if __testing_mode:
        ret_obj = namedtuple('VADER_out', 
                             'tFin tOut colOut presOut eIntOut '+
                             'mBndOut eBndOut mSrcOut eSrcOut '+
                             'userOut '+
                             'nStep nIter nFail residSum iterStep ' +
                             'driverTime advanceTime nextIterTime ' +
                             'userTime')
    else:
        ret_obj = namedtuple('VADER_out', 
                             'tFin tOut colOut presOut eIntOut '+
                             'mBndOut eBndOut mSrcOut eSrcOut '+
                             'userOut nStep nIter nFail')
    if cD['eos_func'] == 0:
        eIntOut = None
    if cD['mass_src_func'] == 0:
        mSrcOut = None
    if cD['mass_src_func'] == 0 and cD['int_en_src_func'] == 0:
        eSrcOut = None
    if nUserOut <= 0:
        userOut = None
    if __testing_mode:
        ret = ret_obj(tFin, tOut, colOut, presOut, eIntOut,
                      mBndOut, eBndOut, mSrcOut, eSrcOut,
                      userOut, nStep.value, nIter.value, nFail.value,
                      residSum, iterStep, driverTime.value, 
                      advanceTime.value, nextIterTime.value,
                      userTime.value)
    else:
        ret = ret_obj(tFin, tOut, colOut, presOut, eIntOut,
                      mBndOut, eBndOut, mSrcOut, eSrcOut, userOut,
                      nStep.value, nIter.value, nFail.value)
    return ret


##################################################################
# Interface to c time step advance routines                      #
##################################################################

def advance(t, dt, grd, col, pres, paramDict, eInt=None,
            no_update=False, c_params=None, **kwargs):
    """
    Interface to the VADER routines to advance one time step.

    Parameters
    ----------
    t : float
       time at start of time step
    dt : float
       size of time step
    grd : class grid
       grid object describing the simulation grid
    col : array
       array of starting column densities
    pres : array
       array of starting pressures
    paramDict : dict
       dict of simulation parameters
    eInt : array
       array of starting internal energies; must be set if gamma or
       delta are specified as c_func instead of constants, ignored
       otherwise
    no_update : bool
       if True, the state quantities will not be altered, and the
       original values will be returned
    c_params : c_void_p
       pointer to parameters to be passed into the c functions
    kwargs : dict
       any additional keywords specified here are appended to the
       paramDict before the simulation is run

    Returns
    -------
    out : nametuple
       out is a namedtuple containing the following output items:
       success : bool
          True if iteration converged
       dtNew : float
          estimated size of next time step; set to -1 if convergence
          failed
       nIter : long
          number of implicit iterations performed
       mBnd : array, shape (2)
          mass transported across inner and outer boundaries during
          this time step; positive values indicate transport in +r
          direction
       eBnd : array, shape (2)
          energy transported across inner and outer boundaries during
          this time step; positive values indicate transport in +r
          direction
       mSrc : array, shape (nr)
          mass column density added to each cell by the source terms
          during this time step; returned only if mass_src is set to
          c_func; otherwise this is None
       eSrc : array, shape (nr)
          energy per unit area added to each cell by the source terms
          during this time step; returned only if mass_src or
          int_en_src is set to c_func; otherwise this is None
       resid : array, shape (max_iter)
          value of the residual after each iteration; returned only if
          the c library is compiled in with -DTESTING_MODE
       rtype : array of uintc, shape (max_iter)
          type of residual (column density, pressure, or internal
          energy) after each iteration; returned only if the c library
          is compiled in with -DTESTING_MODE
       advanceTime : c_double
          total time in seconds spent in advance routine; returned
          only if the code is compiled with -DTESTING_MODE
       nextIterTime : c_double
          total time in seconds spent in getNextIterate routine; returned
          only if the code is compiled with -DTESTING_MODE
       userTime : c_double
          total time in seconds spent in user routines; returned
          only if the code is compiled with -DTESTING_MODE
    """

    loadlib()

    # Get the paramDict ready for passing to c
    cD = cDict(paramDict, **kwargs)

    # Safety check: make sure that we have an eInt value if we're
    # using a complex EOS, and warn if we were given one but we have a
    # simple EOS
    if cD['eos_func'] == 1 and eInt is None:
        raise ValueError("must set eInt if eos_func == 1")
    elif cD['eos_func'] == 0 and eInt is not None:
        print("warning: eInt will not be evolved for EOS with"
              + " constant gamma, delta")

    # eInt must be a numpy array, so create a dummy one if necessary
    if eInt is None:
        eInt = np.zeros(1)

    # Prepare storage for output
    nIter = c_uint(0)
    mBnd = np.zeros(2)
    eBnd = np.zeros(2)
    mSrc = np.zeros(grd.nr)
    eSrc = np.zeros(grd.nr)
    if __testing_mode:
        resid = np.zeros(cD['max_iter'])
        rtype = np.zeros(cD['max_iter'], dtype=np.uintc)
        advanceTime = c_double(0.0)
        nextIterTime = c_double(0.0)
        userTime = c_double(0.0)

    # Decide verbosity
    if cD['verbosity'] > 2:
        verbose = 1
    else:
        verbose = 0

    # Set no_update
    if no_update:
        noUpdate = 1
    else:
        noUpdate = 0

    # Call c function
    if cD['use_BE'] == 0:
        if __testing_mode:
            dtNew = __libvader. \
                    advanceCN(t, dt, grd.c_grd, 
                              col, pres, eInt,
                              mBnd, eBnd, mSrc, eSrc,
                              cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                              cD['alpha_func'], cD['alpha_val'],
                              cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                              cD['ibc_pres_val'], cD['ibc_enth_val'],
                              cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                              cD['obc_pres_val'], cD['obc_enth_val'],
                              cD['mass_src_func'], cD['mass_src_val'],
                              cD['int_en_src_func'], cD['int_en_src_val'],
                              cD['err_tol'], cD['dt_tol'],
                              cD['max_iter'], cD['interp_order'],
                              noUpdate, verbose, grd.c_wksp,
                              c_params, nIter, resid, rtype,
                              advanceTime, nextIterTime, userTime)
        else:
            dtNew = __libvader. \
                    advanceCN(t, dt, grd.c_grd, 
                              col, pres, eInt,
                              mBnd, eBnd, mSrc, eSrc,
                              cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                              cD['alpha_func'], cD['alpha_val'],
                              cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                              cD['ibc_pres_val'], cD['ibc_enth_val'],
                              cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                              cD['obc_pres_val'], cD['obc_enth_val'],
                              cD['mass_src_func'], cD['mass_src_val'],
                              cD['int_en_src_func'], cD['int_en_src_val'],
                              cD['err_tol'], cD['dt_tol'],
                              cD['max_iter'], cD['interp_order'],
                              noUpdate, verbose, grd.c_wksp,
                              c_params, nIter)
    else:
        if __testing_mode:
            dtNew = __libvader. \
                    advanceBE(t, dt, grd.c_grd, 
                              col, pres, eInt,
                              mBnd, eBnd, mSrc, eSrc,
                              cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                              cD['alpha_func'], cD['alpha_val'],
                              cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                              cD['ibc_pres_val'], cD['ibc_enth_val'],
                              cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                              cD['obc_pres_val'], cD['obc_enth_val'],
                              cD['mass_src_func'], cD['mass_src_val'],
                              cD['int_en_src_func'], cD['int_en_src_val'],
                              cD['err_tol'], cD['dt_tol'],
                              cD['max_iter'], cD['interp_order'],
                              noUpdate, verbose, grd.c_wksp,
                              c_params, nIter, resid, rtype,
                              advanceTime, nextIterTime, userTime)
        else:
            dtNew = __libvader. \
                    advanceBE(t, dt, grd.c_grd, 
                              col, pres, eInt,
                              mBnd, eBnd, mSrc, eSrc,
                              cD['eos_func'], cD['gamma_val'], cD['delta_val'],
                              cD['alpha_func'], cD['alpha_val'],
                              cD['ibc_pres'], cD['ibc_enth'], cD['ibc_func'],
                              cD['ibc_pres_val'], cD['ibc_enth_val'],
                              cD['obc_pres'], cD['obc_enth'], cD['obc_func'],
                              cD['obc_pres_val'], cD['obc_enth_val'],
                              cD['mass_src_func'], cD['mass_src_val'],
                              cD['int_en_src_func'], cD['int_en_src_val'],
                              cD['err_tol'], cD['dt_tol'],
                              cD['max_iter'], cD['interp_order'],
                              noUpdate, verbose, grd.c_wksp,
                              c_params, nIter)

    # Construct the object to return, then return it
    if __testing_mode:
        ret_obj = namedtuple('VADER_TEST_advance_out', 
                             'success dtNew nIter mBnd eBnd mSrc eSrc' +
                             ' resid rtype advanceTime nextIterTime' +
                             ' userTime')
    else:
        ret_obj = namedtuple('VADER_advance_out', 
                             'success dtNew nIter mBnd eBnd mSrc eSrc')
    if cD['mass_src_func'] == 0:
        mSrc = None
    if cD['mass_src_func'] == 0 and cD['int_en_src_func'] == 0:
        eSrc = None
    if dtNew < 0:
        success = False
    else:
        success = True
    if __testing_mode:
        ret = ret_obj(success, dtNew, nIter.value, mBnd, eBnd, mSrc, eSrc,
                      resid, rtype, advanceTime.value, nextIterTime.value,
                      userTime.value)
    else:
        ret = ret_obj(success, dtNew, nIter.value, mBnd, eBnd, mSrc, eSrc)
    return ret

