"""
This module defines the grid class, which holds information on the
computational grid.
"""

import numpy as np
import struct
from .interface import gridFree, wkspFree, rotCurveSpline
from ctypes import c_double

# Define some physical constants in CGS units
import scipy.constants as physcons
G = physcons.G*1e3

# Define the grid class
class grid(object):
    """
    A class that contains information on the rotation curve and grid
    layout

    Notes on attributes: arrays without subscripts are all quantities
    measured at cell centers, and have nr elements. Arrays with
    subscript _h are at cell edges, and have nr+1 elements. Arrays
    with subscript _g are at cell centers but with 1 ghost zone, and
    have nr+2 elements.

    Attributes
    ----------
    nr : int
       number of cells
    r, r_h, r_g : array
       Cell center / edge position
    linear : bool
       True for linearly-spaced grids, false for logarithmically
       spaced
    dr, dr_h : array
       dr = r_h[1:] - r_h[:-1]
       dr_h = r_g[1:] - r_g[:-1]
    dlnr, dlnr_h : array
       dlnr = log(r_h[1:]/r_h[:-1])
       dlnr_h = log(r_g[1:]/r_g[:-1])
    area : array
       Cell area
    vphi, vphi_h, vphi_g : array
       Rotation curve velocity
    beta, beta_h, beta_g : array
       Logarithmic derivative of rotation curve with respect to radius
    psi, psi_h, psi_g : array
       Gravitational potential
    psiEff, psiEff_h, psiEff_g : array
       Energy per unit mass for material in circular orbit, equal to 
       psi + vphi^2 / 2
    g_h : array
       A quantity used to computed derivatives;
       g = 2 pi / [vphi (1+beta) dr]        for linear grids
       g = 2 pi / [vphi (1+beta) r d ln r]  for logarithmic grids
    c_grd : c_void_p
       A pointer to a representation of the grid in c; only allocated
       when needed
    c_wksp : c_void_p
       A pointer to a workspace array used for computations in c; only
       allocated when needed
    """

    # Initializer
    def __init__(self, paramDict, rotCurveFun=None, rotCurveTab=None,
                 r_h=None, r_g=None, chk=None, c_grd=None):
        """
        Initializer for class

        Parameters
        ----------
        paramDict : dict
           a dict of parameter values; returned by readParam
        rotCurveFun : callable
           a function to compute the rotation curve; it must take two
           arguments: a numpy array of positions, and the paramDict,
           and return 3 numpy arrays giving the rotation curve
           velocity, the potential, and beta at the input positions
        rotCurveTab : array of shape (2, N)
           a tabulated rotation curve; elements [0,:] give the
           positions, and elements [1,:] give the velocities at those
           positions
        r_h : array
           cell edge positions (for non-uniform grids)
        r_g : array
           cell center positions, including one ghost zone; must be
           set if r_h is set, and vice versa
        chk : file
           a file object pointing to the start of the grid as
           represented in a checkpoint file; if this is set, all other
           inputs are ignored, and the grid is constructed directly
           from the checkpoint
        c_grd : c_grid
           a c_grid object containing a c representation of a grid; if
           this is set, all other inputs are ignored, and the grid is
           built directly from the c representation; it is an error if
           chk and c_grd are both set simultaneously

        Returns
        -------
        Nothing
        """

        # Initialize c data pointers
        self.c_grd = None
        self.c_wksp = None
        
        # If we are reconstructing from a checkpoint or a buffer, just
        # do that and return
        if chk is not None:
            if c_grd is not None:
                raise ValueError("cannot set chk and c_grd" +
                                 " at the same time!")
            self.__init_from_checkpoint__(chk)
            return
        elif c_grd is not None:
            self.__init_from_buffer__(c_grd)
            return
                
        # Compute cell positions and related quantities
        if r_h is None:
            self.nr = paramDict['nr']
            if 'grid_type' in paramDict:
                if paramDict['grid_type'] == 'linear':
                    self.linear = True
                elif paramDict['grid_type'] == 'logarithmic':
                    self.linear = False
                else:
                    raise ValueError("Error: grid_type must be " +
                                     "linear or logarithmic")
            else:
                self.linear = False
            self.dlnr = np.ones(self.nr) * \
                        np.log(paramDict['rmax']/paramDict['rmin']) \
                        / float(self.nr)
            self.dr = np.ones(self.nr) * \
                      (paramDict['rmax'] - paramDict['rmin']) \
                      / float(self.nr)
            self.dlnr_h = np.ones(self.nr+1) * \
                          np.log(paramDict['rmax']/paramDict['rmin']) \
                          / float(self.nr)
            self.dr_h = np.ones(self.nr+1) * \
                        (paramDict['rmax'] - paramDict['rmin']) \
                        / float(self.nr)
            if not self.linear:
                self.r = np.exp(np.log(paramDict['rmin']) + 
                                self.dlnr[0]*(np.arange(self.nr)+0.5))
                self.r_g = np.exp(np.log(paramDict['rmin']) + 
                                  self.dlnr[0]*(np.arange(self.nr+2)-0.5))
                self.r_h = np.exp(np.log(paramDict['rmin']) + 
                                  self.dlnr[0]*(np.arange(self.nr+1)))
            else:
                self.r = paramDict['rmin'] + \
                         (np.arange(self.nr)+0.5)*self.dr[0]
                self.r_g = paramDict['rmin'] + \
                           (np.arange(self.nr+2)-0.5)*self.dr[0]
                self.r_h = paramDict['rmin'] + \
                           (np.arange(self.nr+1))*self.dr[0]
        else:
            self.nr = len(r_h)-1
            self.r_h = np.copy(r_h)
            self.r_g = np.copy(r_g)
            self.r = r_g[1:-1]
            self.dr = self.r_h[1:] - self.r_h[:-1]
            self.dlnr = np.log(self.r_h[1:]/self.r_h[:-1])
            self.dr_h = self.r_g[1:] - self.r_g[:-1]
            self.dlnr_h = np.log(self.r_g[1:]/self.r_g[:-1])
        self.area = np.pi*(self.r_h[1:]**2 - self.r_h[:-1]**2)

        # Compute rotation curve values and their first two logaritmic
        # derivatives
        if not rotCurveFun is None:
            self.vphi, self.psi, self.beta \
                = rotCurveFun(self.r, paramDict)
            self.vphi_h, self.psi_h, self.beta_h \
                = rotCurveFun(self.r_h, paramDict)
            self.vphi_g, self.psi_g, self.beta_g \
                = rotCurveFun(self.r_g, paramDict)
        elif not rotCurveTab is None:
            self.__rotCurveTabInit__(rotCurveTab, paramDict)
        else:
            if not 'rot_curve_type' in paramDict:
                print("Error: no rotation curve specified!\n")
                return None
            if paramDict['rot_curve_type'] == 'keplerian':
                if not 'rot_curve_mass' in paramDict:
                    print("Error: must set rot_curve_mass " +
                          "to use keplerian as rot_curve_type\n")
                    return None
                self.vphi = np.sqrt(G*paramDict['rot_curve_mass'] \
                                    / self.r)
                self.vphi_h = np.sqrt(G*paramDict['rot_curve_mass'] \
                                      / self.r_h)
                self.vphi_g = np.sqrt(G*paramDict['rot_curve_mass'] \
                                      / self.r_g)
                self.beta = np.ones(self.nr) * -0.5
                self.beta_h = np.ones(self.nr+1) * -0.5
                self.beta_g = np.ones(self.nr+2) * -0.5
                self.psi = -G*paramDict['rot_curve_mass']/self.r
                self.psi_h = -G*paramDict['rot_curve_mass']/self.r_h
                self.psi_g = -G*paramDict['rot_curve_mass']/self.r_g
            elif paramDict['rot_curve_type'] == 'flat':
                if not 'rot_curve_velocity' in paramDict:
                    print("Error: must set rot_curve_velocity " +
                          "to use flat as rot_curve_type\n")
                    return None
                self.vphi = np.ones(self.nr) * \
                            paramDict['rot_curve_velocity']
                self.vphi_h = np.ones(self.nr+1) * \
                                paramDict['rot_curve_velocity']
                self.vphi_g = np.ones(self.nr+2) * \
                            paramDict['rot_curve_velocity']
                self.beta = np.zeros(self.nr)
                self.beta_h = np.zeros(self.nr+1)
                self.beta_g = np.zeros(self.nr+2)
                self.psi = np.log(self.r/self.r_h[-1])*self.vphi**2
                self.psi_h = np.log(self.r_h/self.r_h[-1])*self.vphi_h**2
                self.psi_g = np.log(self.r_g/self.r_h[-1])*self.vphi_g**2
            elif paramDict['rot_curve_type'] == 'tabulated':
                if not 'rot_curve_file' in paramDict:
                    print("Error: must set rot_curve_file " +
                          "to use tabulated as rot_curve_type\n")
                    return None
                rvTab = self.__rotCurveTabRead__(paramDict['rot_curve_file'])
                self.__rotCurveTabInit__(rvTab, paramDict)
            else:
                print("Error: unknown rot_curve_type: " +
                      paramDict['rot_curve_type'] + "\n")
                return None


        # Utility quantities

        # psiEff = vphi^2/2 + psi
        self.psiEff = self.vphi**2/2 + self.psi
        self.psiEff_h = self.vphi_h**2/2 + self.psi_h
        self.psiEff_g = self.vphi_g**2/2 + self.psi_g

        # g = (2 pi / vphi) [ (1-beta)/(1+beta) ] (1 / dx), where
        # dx = dr for linear grid, dx = r dlnr for logarithmic grid
        if self.linear:
            self.g_h = 2.0*np.pi \
                       / (self.dr_h * self.vphi_h * (1+self.beta_h))
        else:
            self.g_h = 2.0*np.pi \
                       / (self.dlnr_h * self.r_h * self.vphi_h * 
                          (1+self.beta_h))

        # Placeholders for pointers to c memory
        self.c_grd = None
        self.c_wksp = None

    # Destructor
    def __del__(self):
        if self.c_grd is not None: gridFree(self.c_grd)
        if self.c_wksp is not None: wkspFree(self.c_wksp)

    # Function to initialize from a buffer
    def __init_from_buffer__(self, c_grd):

        # No workspace allocated
        self.c_wksp = None

        # Store the buffer
        self.c_grd = c_grd

        # Grab nr and linear from the buffer
        self.nr = int(c_grd.nr)
        self.linear = c_grd.linear
        self.r_g = np.array(c_grd.r_g[:self.nr+2])
        self.r_h = np.array(c_grd.r_h[:self.nr+1])
        self.dr_g = np.array(c_grd.dr_g[:self.nr+2])
        self.area = np.array(c_grd.area[:self.nr])
        self.vphi_g = np.array(c_grd.vphi_g[:self.nr+2])
        self.vphi_h = np.array(c_grd.vphi_h[:self.nr+1])
        self.psiEff_g = np.array(c_grd.psiEff_g[:self.nr+2])
        self.psiEff_h = np.array(c_grd.psiEff_h[:self.nr+1])
        self.g_h = np.array(c_grd.g_h[:self.nr+1])

        # Compute derived quantities
        self.r = self.r_g[1:-1]
        self.dr = self.r_h[1:] - self.r_h[:-1]
        self.dlnr = np.log(self.r_h[1:]/self.r_h[:-1])
        self.dr_h = self.r_g[1:] - self.r_g[:-1]
        self.dlnr_h = np.log(self.r_g[1:]/self.r_g[:-1])

        
    # Function to initialize from a checkpoint
    def __init_from_checkpoint__(self, chk):

        # Read size
        data = chk.read(struct.calcsize('L'))
        self.nr, = struct.unpack('L', data)
        # Read if grid is linear
        data = chk.read(struct.calcsize('?'))
        self.linear, = struct.unpack('?', data)
        # Read grid data
        data = chk.read(struct.calcsize('d'*(self.nr+2)))
        self.r_g = np.array(struct.unpack('d'*(self.nr+2), data))
        data = chk.read(struct.calcsize('d'*(self.nr+1)))
        self.r_h = np.array(struct.unpack('d'*(self.nr+1), data))
        data = chk.read(struct.calcsize('d'*(self.nr+2)))
        self.dr_g = np.array(struct.unpack('d'*(self.nr+2), data))
        data = chk.read(struct.calcsize('d'*self.nr))
        self.area = np.array(struct.unpack('d'*self.nr, data))
        data = chk.read(struct.calcsize('d'*(self.nr+2)))
        self.vphi_g = np.array(struct.unpack('d'*(self.nr+2), data))
        data = chk.read(struct.calcsize('d'*(self.nr+1)))
        self.vphi_h = np.array(struct.unpack('d'*(self.nr+1), data))
        data = chk.read(struct.calcsize('d'*(self.nr+2)))
        self.beta_g = np.array(struct.unpack('d'*(self.nr+2), data))
        data = chk.read(struct.calcsize('d'*(self.nr+1)))
        self.beta_h = np.array(struct.unpack('d'*(self.nr+1), data))
        data = chk.read(struct.calcsize('d'*(self.nr+2)))
        self.psiEff_g = np.array(struct.unpack('d'*(self.nr+2), data))
        data = chk.read(struct.calcsize('d'*(self.nr+1)))
        self.psiEff_h = np.array(struct.unpack('d'*(self.nr+1), data))
        data = chk.read(struct.calcsize('d'*(self.nr+1)))
        self.g_h = np.array(struct.unpack('d'*(self.nr+1), data))

        # Compute derived quantities
        self.r = self.r_g[1:-1]
        self.dr = self.r_h[1:] - self.r_h[:-1]
        self.dlnr = np.log(self.r_h[1:]/self.r_h[:-1])
        self.dr_h = self.r_g[1:] - self.r_g[:-1]
        self.dlnr_h = np.log(self.r_g[1:]/self.r_g[:-1])
        self.vphi = self.vphi_g[1:-1]
        self.beta = self.beta_g[1:-1]
        self.psiEff = self.psiEff_g[1:-1]

        # Placeholders for pointers to c memory
        self.c_grd = None
        self.c_wksp = None
        
    # Function to read in a tabulated rotation curve
    def __rotCurveTabRead__(self, fileName):

        fp = open(fileName, 'r')
        rIn = []
        vIn = []
        for line in fp:
            # Skip lines that are blank or comments
            lstrip = line.strip()
            if len(lstrip) == 0:
                continue
            if lstrip[0] == '#':
                continue
            # Parse entry
            lsplit = lstrip.split()
            rIn.append(float(lsplit[0]))
            vIn.append(float(lsplit[1]))
        fp.close()
        rvTab = np.zeros((2, len(rIn)))
        rvTab[0,:] = rIn
        rvTab[1,:] = vIn
        return rvTab


    # Function to do tabulated rotation curve initialization. Since the
    # bspline capablity we want is in the GSL, we do this by parsing the
    # file to get the entries, then calling some c code to do the
    # computations and interface with GSL.
    def __rotCurveTabInit__(self, rvTab, paramDict):

        # Step 1: read parameters from dict
        if 'bspline_degree' in paramDict:
            bsplineDegree = paramDict['bspline_degree']
        else:
            bsplineDegree = 6
        if 'bspline_breakpoints' in paramDict:
            bsplineBreakpoints = paramDict['bspline_breakpoints']
        else:
            bsplineBreakpoints = 15

        # Step 2: safety check: make sure grid is entirely within
        # tabulated grid
        if (np.amin(self.r_g) < np.amin(rvTab[0,:])) or \
           (np.amax(self.r_g) > np.amax(rvTab[0,:])):
            raise ValueError(
                ("Error: entire computational mesh must fall within "
                 + "tabulated rotation curve; mesh range = {:e} to "
                 + "{:e}, table range = {:e} to {:e}\n").
                format(np.amin(self.r_g), np.amax(self.r_g),
                       np.amin(rvTab[0,:]), np.amax(rvTab[0,:])))

        # Step 3: create array of all radial positions
        rAll = np.zeros(2*self.nr+3)
        rAll[::2] = self.r_g
        rAll[1::2] = self.r_h

        # Step 4: do interpolation
        vphi, psi, beta \
            = rotCurveSpline(rvTab[0,:], rvTab[1,:], rAll, 
                             bsplineDegree, bsplineBreakpoints)

        # Step 5: unpack results
        self.vphi = np.copy(vphi[2:-1:2])
        self.vphi_h = np.copy(vphi[1::2])
        self.vphi_g = np.copy(vphi[::2])
        self.psi = np.copy(psi[2:-1:2])
        self.psi_h = np.copy(psi[1::2])
        self.psi_g = np.copy(psi[::2])
        self.beta = np.copy(beta[2:-1:2])
        self.beta_h = np.copy(beta[1::2])
        self.beta_g = np.copy(beta[::2])

