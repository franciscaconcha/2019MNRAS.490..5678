from amuse.lab import *
import numpy
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from matplotlib import pyplot
import gzip
import copy
from scipy import interpolate
from decorators import timer


def column_density(r):
    rd = 400 | units.AU
    rc = 400 | units.AU
    Md = 1 | units.MJupiter

    #if r < rd:
    #    return 1E-12

    Sigma_0 = Md / (2 * numpy.pi * rc**2 * (1 - numpy.exp(-rd/rc)))
    Sigma = Sigma_0 * (r/rc) * numpy.exp(-r/rc)
    return Sigma


def disk_radius(disk, density_limit):
    """ Calculate the radius of a disk in a vader grid.

    :param disk: vader code of disk
    :param density_limit: density limit to designate disk border
    :return: disk radius in units.AU
    """
    prev_r = disk.grid.r[0]

    for cell, r in zip(disk.grid.column_density, disk.grid.r):
        if cell.value_in(units.g / units.cm**2) < density_limit:
            return prev_r
        prev_r = r


def disk_mass(disk, radius):
    """ Calculate the mass of a vader disk inside a certain radius.

    :param disk: vader code of disk
    :param radius: disk radius to consider for mass calculation
    :return: disk mass in units.MJupiter
    """
    mass_cells = disk.grid.r[disk.grid.r <= radius]
    total_mass = 0

    for m, d, a in zip(mass_cells, disk.grid.column_density, disk.grid.area):
        total_mass += d.value_in(units.MJupiter / units.cm**2) * a.value_in(units.cm**2)

    return total_mass | units.MJupiter


@timer
def main():
    disk = vader(redirection='none')
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        100,  # Number of cells
        False,  # Linear?
        0.5 | units.AU,  # Rmin
        5000 | units.AU,  # Rmax
        1 | units.MJupiter  # Mass
    )

    #disk.parameters.verbosity = 1

    sigma = column_density(disk.grid.r)
    disk.grid.column_density = sigma

    # The pressure follows the ideal gas law with a mean molecular weight of 2.33 hydrogen masses.
    # Since gamma ~ 1, this should not influence the simulation
    mH = 1.008 * constants.u
    T = 100. | units.K
    disk.grid.pressure = sigma * constants.kB * T / (2.33 * mH)

    """print((disk.grid.area * disk.grid.column_density).sum())
    print disk.grid.r
    less = [disk.grid.r > 10E13 | units.cm]# = 1E-8 | units.g / (units.cm)**2
    print less
    #disk.grid.column_density[less] = 1
    for x in range(len(disk.grid.column_density)):
        if disk.grid.r[x] > 10E13 | units.cm:
            print "yes"
            disk.grid[x].column_density = 1E-12 | units.g / (units.cm)**2.0"""

    disk.evolve_model(0.04 | units.Myr)
    print(disk.grid.column_density)
    r = disk_radius(disk, 1E-4)
    disk_mass(disk, r)
    #pyplot.plot(numpy.array(disk.grid.r.value_in(units.AU)), numpy.array(disk.grid.column_density.value_in(units.g / (units.cm)**2)))
    #pyplot.show()


    disk.stop()


if __name__ == '__main__':
    main()
