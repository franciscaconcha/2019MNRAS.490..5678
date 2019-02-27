from amuse.lab import *
import numpy
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from matplotlib import pyplot
import gzip
import copy
from scipy import interpolate
from decorators import timer
import os

from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
mpl.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
mpl.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels

# Had to do this for now as a workaround, will try to get rid of it soon
global plot_colors
plot_colors = {"gas": "#ca5670", "no_gas": "#638ccc", "gas_expulsion": "#72a555"}

def column_density(grid, r0, mass, lower_density=1E-12 | units.g / units.cm**2):
    r = grid.value_in(units.AU) | units.AU
    rd = r0
    Md = mass

    Sigma_0 = Md / (2 * numpy.pi * r0 ** 2 * (1 - numpy.exp(-rd / r0)))
    Sigma = Sigma_0 * (r0 / r) * numpy.exp(-r / r0) * (r <= r0) + lower_density
    return Sigma


def get_disk_radius(disk, density_limit=1E-11):
    """ Calculate the radius of a disk in a vader grid.

    :param disk: Disk to calculate radius on.
    :param density_limit: Density limit to designate disk border.
    :return: Disk radius in units.AU
    """
    prev_r = disk.grid.r[0]

    for cell, r in zip(disk.grid.column_density, disk.grid.r):
        if cell.value_in(units.g / units.cm**2) <= density_limit:
            return prev_r.value_in(units.AU) | units.AU
        prev_r = r

    return prev_r.value_in(units.AU) | units.AU


def get_disk_mass(disk, radius):
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

def initialize_vader_code(disk_radius, disk_mass, alpha, r_min=0.05 | units.AU, r_max=2000 | units.AU, n_cells=50, linear=True):
    """ Initialize vader code for given parameters.

    :param disk_radius: disk radius. Must have units.Au
    :param disk_mass: disk mass. Must have units.MSun
    :param alpha: turbulence parameter for viscosity
    :param r_min: minimum radius of vader grid. Must have units.AU
    :param r_max: maximum radius of vader grid. Must have units.AU
    :param n_cells: number of cells for vader grid
    :param linear: linear interpolation
    :return: instance of vader code
    """
    disk = vader(redirection='none')
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        n_cells,  # Number of cells
        linear,  # Linear?
        r_min,  # Grid Rmin
        r_max,  # Grid Rmax
        disk_mass  # Disk mass
    )

    #disk.parameters.verbosity = 1

    sigma = column_density(disk.grid.r, disk_radius, disk_mass)
    disk.grid.column_density = sigma

    # The pressure follows the ideal gas law with a mean molecular weight of 2.33 hydrogen masses.
    # Since gamma ~ 1, this should not influence the simulation
    mH = 1.008 * constants.u
    T = 100. | units.K
    disk.grid.pressure = sigma * constants.kB * T / (2.33 * mH)

    disk.parameters.inner_pressure_boundary_type = 1
    #disk.parameters.inner_boundary_function = True
    disk.parameters.inner_pressure_boundary_torque = 0.0 | units.g * units.cm ** 2 / units.s ** 2
    disk.parameters.alpha = alpha
    disk.parameters.maximum_tolerated_change = 1E99
    disk.set_parameter(0, False)  # Disk parameter for non-convergence. True: disk diverged

    return disk


@timer
def main():
    cells = [10, 100, 250, 500]
    rout = [2500, 5000] | units.au

    t_end = 10 | units.Myr
    profiles = {}
    radii = {}

    path = "tests_results"
    try:
        os.makedirs(path)
        print "Results path created"
    except OSError, e:
        if e.errno != 17:
            raise
        # time.sleep might help here
        pass

    for c in cells:
        profiles[c] = []
        radii[c] = []
        for r in rout:
            disk = initialize_vader_code(100 | units.au, 0.1 | units.MSun, 5E-3, r_max=r, n_cells=c, linear=False)
            disk.evolve_model(t_end)
            p = disk.grid.column_density
            r = disk.grid.r
            disk.stop()
            profiles[c].append(p)
            radii[c].append(r)

    for key, val in profiles.items():
        name = "{0}/{1}cells.txt".format(path, key)
        v0 = val[0].value_in(units.g / units.cm ** 2)
        v1 = val[1].value_in(units.g / units.cm ** 2)
        numpy.savetxt(name, numpy.transpose([v0, v1]))

    for c in cells:
        r_2500 = radii[c][0].value_in(units.au)
        r_5000 = radii[c][1].value_in(units.au)
        d_2500 = profiles[c][0].value_in(units.g / units.cm ** 2)
        d_5000 = profiles[c][1].value_in(units.g / units.cm ** 2)

        fig = pyplot.figure(figsize=(12, 8))
        ax = pyplot.gca()
        #fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True)
        pyplot.xlabel("Disk radius [au]")
        pyplot.ylabel("Column density [g / cm**2]")
        pyplot.title("{0} cells".format(c))

        ax.plot(r_2500, d_2500, lw=3, label='2500 au')
        ax.plot(r_5000, d_5000, lw=3, label='5000 au')
        pyplot.legend(loc='upper right')

        pyplot.savefig('{0}/Density_{1}cells.png'.format(path, c))




if __name__ == '__main__':
    main()
