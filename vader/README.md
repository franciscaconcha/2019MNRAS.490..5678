# README #

### What is VADER? ###

VADER stands for Viscous Accretion Disk Evolution Resource. VADER is a flexible, general code intended to simulate the time evolution of thin axisymmetric accretion disks in time-steady potentials. VADER can handle arbitrary viscosities, equations of state, boundary conditions, and source and sink terms for both mass and energy. The physical system that the code models, and the equations that it solves, are fully described in [Krumholz & Forbes, 2015, Astronomy and Computing, 11, 1](http://adsabs.harvard.edu/abs/2015A%26C....11....1K).

### Layout of this repository ###

The VADER repository contains a series of subdirectories whose contents are as follows.

* *bin*: this contains VADER binaries when they are compiled, either as executables or shared object code
* *doc*: contains detailed code documentation
* *output*: contains outputs of the test suite
* *test*: contains Python scripts to run several test problems.
* *vader*: this contains Python routines that wrap the c code. When VADER is compiled in dynamically-linked library mode (see below), the routines make it possible to execute VADER simulations from Python programs. They also provide utilities for parsing input files and similar tasks. Functions to run particular standard tests are in the vader/test subdirectory.
* *vader_csrc*: this contains the main c source code for VADER. General routines are in this directory, while problem-specific routines (e.g. those giving the functional form of the viscosity) are in the vader_csrc/prob/ directory.

### Installing and configuring ###

#### Dependencies ####

* The core c routines require the [GNU scientific library](http://www.gnu.org/software/gsl/) version >= 1.12.
* The Python wrapper routines require Python >= 2.6 (3.x also ok)
* The Python wrapper routines require numPy >= 1.6.1
* The Python wrapper routines require sciPy >= 0.7.0


#### Compiling ####

VADER comes with a basic Makefile, in vader_csrc, that should be able to compile the code in most sane unix-based build environments, where CC and MAKE are defined in the environment or default to reasonable values. This Makefile *may* work for windows also, but it has not been tested in a windows environment.

The Makefile can be edited to manually specify choices of compiler, linker, etc. The GSL library headers must be included in C_INCLUDE_PATH, and the GSL library shared object files must be included in LD_LIBRARY_PATH.

VADER can be built in two modes: standalone executable mode and dynamically-linked library mode. To build a standalone executable, from the main vader directory simply do

`make exec PROB=probname`

where probname is the name of a problem to be compiled; problems consist of custom definitions of functions describing the viscosity, equation of state, boundary conditions, and source and sink terms. See the user manual for details on defining your own problem. Test problems included with the library are `selfsim`, `ring`, `gidisk`, and `ringrad`. Upon successful compilation, an executable `vader.ex` should appear in the bin directory.

To build in dynamically-linked library mode, suitable for calling from Python, from the main vader directory do

`make lib PROB=probname`

where probname is again the problem to be compiled. For dynamically-linked library mode, `PROB=probname` can be omitted, in which case the problem is set to `none`. This problem implements no custom functions for viscosity, equation of state, boundary conditions, and source and sink terms, but simulations with this problem can still be carried out by specifying these quantities by constants instead of functions. Again, see the user manual for details. Upon successful compilation in dynamically-linked library mode, a dynamic library object called `libvader.x` should appear in the bin directory, where the extension will be the appropriate extension name for dynamically-linked libraries on the platform used to compile the code.

### Running the test suite ###

Several automated test scripts are provided, in the test directory. The tests provided are:

* *selfsim.py*: evolution of a self-similar viscous disk
* *selfsim_resolution.py*: convergence study on evolution of a self-similar viscous disk
* *ring.py*: evolution of an initially-singular ring
* *gidisk.py*: a series of simulations of a gravitational instability-dominated disk
* *ringrad.py*: evolution of an initially-singular ring including a complex equation of state with significant contributions from both radiation and gas pressure.
* *rotcurvefit.py*: tests the tabulated rotation curve fitting capability against both an analytic function and some noisy data
* *performance.py*: runs all the test problems with the code compiled in testing mode (see the manual), and reports the performance results.

Each of these tests may be run by typing

`python test/TESTNAME.py`

in the main directory, where TESTNAME is the name of the test script to be run. The script will automatically configure and compile the library for the appropriate test problem. Each test script writes out one or more figures .pdf format to the output directory. Benchmark results are stored in the repository version of the output directory, so users can check the impact of any changes they make to the code by running one of the test problems and comparing to the repository benchmarks.


### Questions, bugs, and getting involved ###

If you have questions about VADER, discovered any bugs, or want to contribute to ongoing development, please contact [Mark Krumholz](https://sites.google.com/a/ucsc.edu/krumholz/), mkrumhol@ucsc.edu.