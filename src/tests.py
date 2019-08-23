""" To reproduce the tests and figures of the appendix """

from amuse.lab import *
import numpy
from matplotlib import pyplot
from decorators import timer
import time

from matplotlib import rc
import matplotlib as mpl

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
mpl.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
mpl.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels


def column_density(grid,
                   rc,
                   mass,
                   lower_density=1E-12 | units.g / units.cm**2):
    """ Disk column density definition as in Eqs. 1, 2, and 3 of the paper.
        (Lynden-Bell & Pringle, 1974: Anderson et al. 2013)

    :param grid: disk grid
    :param rc: characteristic disk radius
    :param mass: disk mass
    :param lower_density: density limit for defining disk edge
    :return: disk column density in g / cm**2
    """
    r = grid.value_in(units.au) | units.au
    rd = rc  # Anderson et al. 2013
    Md = mass

    Sigma_0 = Md / (2 * numpy.pi * rc ** 2 * (1 - numpy.exp(-rd / rc)))
    Sigma = Sigma_0 * (rc / r) * numpy.exp(-r / rc) * (r <= rc) + lower_density
    return Sigma


def get_disk_radius(disk,
                    density_limit=1E-10):
    """ Calculate the radius of a disk in a vader grid.

    :param disk: vader disk
    :param density_limit: density limit to designate disk border
    :return: disk radius in units.au
    """
    prev_r = disk.grid[0].r

    for i in range(len(disk.grid.r)):
        cell_density = disk.grid[i].column_density.value_in(units.g / units.cm ** 2)
        if cell_density < density_limit:
            return prev_r.value_in(units.au) | units.au
        prev_r = disk.grid[i].r

    return prev_r.value_in(units.au) | units.au


def get_disk_mass(disk,
                  radius):
    """ Calculate the mass of a vader disk inside a certain radius.

    :param disk: vader disk
    :param radius: disk radius to consider for mass calculation
    :return: disk mass in units.MJupiter
    """
    mass_cells = disk.grid.r[disk.grid.r <= radius]
    total_mass = 0

    for m, d, a in zip(mass_cells, disk.grid.column_density, disk.grid.area):
        total_mass += d.value_in(units.MJupiter / units.cm**2) * a.value_in(units.cm**2)

    return total_mass | units.MJupiter


def initialize_vader_code(disk_radius,
                          disk_mass,
                          alpha,
                          r_min=0.05 | units.au,
                          r_max=2000 | units.au,
                          n_cells=100,
                          linear=True):
    """ Initialize vader code for given parameters.

    :param disk_radius: disk radius. Must have units.au
    :param disk_mass: disk mass. Must have units.MSun
    :param alpha: turbulence parameter for viscosity, adimensional
    :param r_min: minimum radius of vader grid. Must have units.au
    :param r_max: maximum radius of vader grid. Must have units.au
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
    disk.parameters.inner_pressure_boundary_torque = 0.0 | units.g * units.cm ** 2 / units.s ** 2
    disk.parameters.alpha = alpha
    disk.parameters.maximum_tolerated_change = 1E99
    global diverged_disks
    diverged_disks[disk] = False

    return disk


@timer
def main():
    cells = [10, 50, 100, 150, 200, 250]
    rout = 5000 | units.au

    profiles = {}
    areas = {}
    grids = {}
    disk_radii = {}

    times = []
    t = 0 | units.Myr
    dt = 0.001 | units.Myr
    t_end = 1 | units.Myr

    while t < t_end:
        t += dt
        times.append(t.value_in(units.Myr))

    c_times = []

    for c in cells:
        start_time = time.ctime()
        print "Starting run for {0} cells at {1}".format(c, start_time)
        start_time = time.time()

        profiles[c] = []
        areas[c] = []
        grids[c] = []
        disk_radii[c] = []

        disk = initialize_vader_code(100 | units.au, 0.1 | units.MSun, 1E-4, r_max=rout, n_cells=c, linear=False)
        t = 0 | units.Myr
        radius = []

        while t < t_end:
           t += dt
           disk.evolve_model(t)
           radius.append(get_disk_radius(disk, density_limit=1E-8).value_in(units.au))
        
        p = disk.grid.column_density
        a = disk.grid.area
        r = disk.grid.r

        disk.stop()

        profiles[c].append(p)
        areas[c].append(a)
        grids[c].append(r)
        disk_radii[c].append(radius)

        elapsed = int(time.time() - start_time)
        print "Ended."
        print 'ELAPSED TIME = {:02d}:{:02d}:{:02d}'.format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60)

        minutes = elapsed % 3600 // 60
        c_times.append(minutes)

    avr = []

    # Figure A1
    for r in cells:
        if r > 50:
            avr.append(disk_radii[r][0])
        pyplot.plot(times,
                    disk_radii[r][0],
                    label='{0} cells'.format(r), lw=2)

    pyplot.legend()
    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel('Disk radius [au]')
    #pyplot.savefig('radii.png')
    pyplot.show()

    # Figure A2
    for p in cells:
        profile = profiles[p][0].value_in(units.MJupiter / units.cm**2)
        area = areas[p][0].value_in(units.cm**2)

        prev_mass = 0.0
        mass = []

        for i in range(len(profile)):
            prev_mass += profile[i] * area[i]
            mass.append(prev_mass)

        pyplot.plot(grids[p][0].value_in(units.au),
                    mass,
                    label='{0} cells'.format(p), lw=2)

    pyplot.xlim([0, 50])
    pyplot.ylim([0, 150])
    pyplot.xlabel('Disk radius [au]')
    pyplot.ylabel('Cumulative mass [$M_{Jup}$]')
    pyplot.legend()
    #pyplot.savefig('cumulative_mass.png')
    pyplot.show()

    # Figure A3
    for p in cells:
        pyplot.loglog(grids[p][0].value_in(units.au),
                      profiles[p][0].value_in(units.g / units.cm**2),
                      label='{0} cells'.format(p))
    pyplot.legend()
    pyplot.xlim([0, 1000])
    pyplot.xlabel('Disk radius [au]')
    pyplot.ylabel('Surface density [g / cm$^2$]')
    #pyplot.savefig('profiles.png')
    pyplot.show()


if __name__ == '__main__':
    main()
