from amuse.lab import *
import numpy
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from matplotlib import pyplot
import gzip
import copy
from scipy import interpolate


def column_density(r):
    rd = 0.1 | units.AU
    rc = 10 | units.AU
    Md = 1 | units.MSun

    #if r < rd:
    #    return 1E-12

    Sigma_0 = Md / (2 * numpy.pi * rc**2 * (1 - numpy.exp(-rd/rc)))
    Sigma = Sigma_0 * (r/rc) * numpy.exp(-r/rc)
    return Sigma


def main():
    disk = vader(redirection='none')
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        128,  # Number of cells
        True,  # Linear?
        0.1 | units.AU,  # Rmin
        30 | units.AU,  # Rmax
        1 | units.MSun  # Mass
    )

    #disk.parameters.verbosity = 1

    sigma = column_density(disk.grid.r)
    disk.grid.column_density = sigma

    # The pressure follows the ideal gas law with a mean molecular weight of 2.33 hydrogen masses.
    # Since gamma ~ 1, this should not influence the simulation
    mH = 1.008 * constants.u
    T = 100. | units.K
    disk.grid.pressure = sigma * constants.kB * T / (2.33 * mH)

    print((disk.grid.area * disk.grid.column_density).sum())
    print disk.grid.r.value_in(units.AU)
    print disk.grid.column_density.shape

    import matplotlib.pyplot

    pyplot.plot(numpy.array(disk.grid.r.value_in(units.AU)), numpy.array(disk.grid.column_density.value_in(units.g / (units.cm)**2)))


    """less = [disk.grid.r > 10E13 | units.cm]# = 1E-8 | units.g / (units.cm)**2
    print less
    #disk.grid.column_density[less] = 1
    for x in range(len(disk.grid.column_density)):
        if disk.grid.r[x] > 10E13 | units.cm:
            print "yes"
            disk.grid[x].column_density = 1E-12 | units.g / (units.cm)**2.0
    print(disk.grid.column_density)"""
    disk.evolve_model(0.1 | units.Myr)
    pyplot.plot(numpy.array(disk.grid.r.value_in(units.AU)), numpy.array(disk.grid.column_density.value_in(units.g / (units.cm)**2)))
    pyplot.show()
    disk.stop()


if __name__ == '__main__':
    main()
