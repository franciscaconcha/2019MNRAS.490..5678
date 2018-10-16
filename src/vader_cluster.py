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


def initialize_vader_code(r_min, r_max, disk_mass, n_cells=128, linear=True):
    """ Initialize vader code for given parameters.

    :param r_min: minimum radius of disk. Must have units.AU
    :param r_max: (maximum) radius of disk. Must have units.AU
    :param disk_mass: disk mass. Must have units.MSun
    :param n_cells: number of cells for grid
    :param linear: linear interpolation
    :return: instance of vader code
    """
    disk = vader(redirection='none')
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        n_cells,  # Number of cells
        linear,  # Linear?
        r_min,  # Rmin
        r_max,  # Rmax
        disk_mass  # Mass
    )

    disk.parameters.verbosity = 1

    sigma = column_density(disk.grid.r)
    disk.grid.column_density = sigma

    # The pressure follows the ideal gas law with a mean molecular weight of 2.33 hydrogen masses.
    # Since gamma ~ 1, this should not influence the simulation
    mH = 1.008 * constants.u
    T = 100. | units.K
    disk.grid.pressure = sigma * constants.kB * T / (2.33 * mH)

    return disk


def main(N, Rvir, Qvir, alpha, R, gas_presence, gas_expulsion, gas_expulsion_onset, gas_expulsion_timescale,
         t_ini, t_end, save_interval, run_number, save_path,
         gamma=1,
         mass_factor_exponent=0.2,
         truncation_parameter=1. / 3,
         gas_to_stars_mass_ratio=2.0,
         gas_to_stars_plummer_radius_ratio=1.0,
         plummer_radius=0.5 | units.parsec,
         dt=2000 | units.yr,
         temp_profile=0.5,
         Rref=1.0 | units.AU,
         Tref=280 | units.K,
         mu=2.3 | units.g / units.mol,
         filename=''):

    t_end = t_end | units.Myr

    max_stellar_mass = 100 | units.MSun
    stellar_masses = new_kroupa_mass_distribution(N, max_stellar_mass)  # , random=False)
    converter = nbody_system.nbody_to_si(stellar_masses.sum(), Rvir)
    stars = new_plummer_model(N, converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    stars.stellar_mass = stellar_masses

    # Bright stars: no disks; emit FUV radiation
    #bright_stars = [s for s in stars if s.stellar_mass.value_in(units.MSun) > 1.9]
    bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

    # Small stars: with disks; radiation not considered
    #small_stars = [s for s in stars if s.stellar_mass.value_in(units.MSun) < 1.9]
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) < 1.9]

    bright_stars.disk_mass = 0 | units.MSun
    small_stars.disk_mass = 0.1 * small_stars.stellar_mass

    print bright_stars.disk_mass
    print small_stars.disk_mass


    """disk_masses = 0.1 * stellar_masses
    converter = nbody_system.nbody_to_si(stellar_masses.sum() + disk_masses.sum(), Rvir)

    stars = new_plummer_model(N, converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    print("stellar masses: ", stellar_masses)

    stars.mass = stellar_masses
    stars.initial_characteristic_disk_radius = 30 * (stars.mass.value_in(units.MSun) ** 0.5) | units.AU
    stars.disk_radius = stars.initial_characteristic_disk_radius
    print stars.disk_radius
    stars.initial_disk_mass = disk_masses
    stars.disk_mass = stars.initial_disk_mass
    stars.total_star_mass = stars.initial_disk_mass + stars.mass
    stars.viscous_timescale = viscous_timescale(stars, alpha, temp_profile, Rref, Tref, mu, gamma)
    stars.last_encounter = 0.0 | units.yr

    # Bright star
    stars[2].mass = 5 | units.MSun
    stars[2].initial_characteristic_disk_radius = 0 | units.AU
    stars[2].initial_disk_mass = 0 | units.MSun
    stars[2].total_star_mass = stars[2].mass
    stars[2].viscous_timescale = 0 | units.yr

    # print("star.mass: ", stars.mass)
    # print("star.mass: ", stars.mass.value_in(units.MSun))

    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    # stellar.particles.add_particles(Particles(mass=stars.stellar_mass))
    stellar.particles.add_particles(stars)

    print("star.mass: ", stars.mass.value_in(units.MSun))

    initial_luminosity = stellar.particles.luminosity
    stars.luminosity = initial_luminosity
    stars.temperature = stellar.particles.temperature
    dt = 5 | units.Myr

    print("L(t=0 Myr) = {0}".format(initial_luminosity))
    print("R(t=0 Myr) = {0}".format(stellar.particles.radius.in_(units.RSun)))
    print("m(t=0 Myr) = {0}".format(stellar.particles.mass.in_(units.MSun)))
    print("Temp(t=0 Myr) = {0}".format(stellar.particles[2].temperature.in_(units.K)))

    channel_to_framework = stellar.particles.new_channel_to(stars)
    write_set_to_file(stars, 'results/0.hdf5', 'amuse')

    # temp, temp2 = [], []
    lower_limit, upper_limit = 1000, 3000  # Limits for FUV, in Angstrom
    fuv_filename_base = "p00/t{0}g{1}p00k2.flx.gz"
    g = "00"

    gravity = ph4(converter)
    gravity.parameters.timestep_parameter = 0.01
    gravity.parameters.epsilon_squared = (100 | units.AU) ** 2
    gravity.particles.add_particles(stars)

    channel_from_stellar_to_framework \
        = stellar.particles.new_channel_to(stars)
    channel_from_stellar_to_gravity \
        = stellar.particles.new_channel_to(gravity.particles)
    channel_from_gravity_to_framework \
        = gravity.particles.new_channel_to(stars)

    Etot_init = gravity.kinetic_energy + gravity.potential_energy
    dE_gr = 0 | Etot_init.unit
    time = 0.0 | t_end.unit
    dt = stellar.particles.time_step.amin()

    # Bright stars: no disks; emit FUV radiation
    bright_stars = [s for s in stars if s.mass.value_in(units.MSun) > 3]

    # Small stars: with disks; radiation not considered
    small_stars = [s for s in stars if s.mass.value_in(units.MSun) < 3]

    print "INIT:"
    print stars.x
    print stars.y
    print stars.z
    initx, inity, initz = copy.deepcopy(stars.x), copy.deepcopy(stars.y), copy.deepcopy(stars.z)

    # Read FRIED grid
    grid = numpy.loadtxt('friedgrid.dat', skiprows=2)

    # Getting only the useful parameters from the grid (not including Mdot)
    FRIED_grid = grid[:, [0, 1, 2, 4]]
    grid_log10Mdot = grid[:, 5]

    grid_stellar_masses = FRIED_grid[:, 0]
    grid_FUV = FRIED_grid[:, 1]
    grid_disk_mass = FRIED_grid[:, 2]
    grid_disk_radius = FRIED_grid[:, 3]"""

    """print((disk.grid.area * disk.grid.column_density).sum())
    print disk.grid.r
    less = [disk.grid.r > 10E13 | units.cm]# = 1E-8 | units.g / (units.cm)**2
    print less
    #disk.grid.column_density[less] = 1
    for x in range(len(disk.grid.column_density)):
        if disk.grid.r[x] > 10E13 | units.cm:
            print "yes"
            disk.grid[x].column_density = 1E-12 | units.g / (units.cm)**2.0
    print(disk.grid.column_density)
    disk.evolve_model(0.04 | units.Myr)
    #print(disk.grid.mass_source_difference)
    print((disk.grid.area * disk.grid.column_density).sum())
    disk.stop()"""

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation parameters
    result.add_option("-n", dest="run_number", type="int", default=0,
                      help="run number [%default]")
    result.add_option("-s", dest="save_path", type="string", default='.',
                      help="path to save the results [%default]")
    result.add_option("-i", dest="save_interval", type="int", default=50000 | units.yr,
                      help="time interval of saving a snapshot of the cluster [%default]")

    # Cluster parameters
    result.add_option("-N", dest="N", type="int", default=2000,
                      help="number of stars [%default]")
    result.add_option("-R", dest="Rvir", type="float",
                      unit=units.parsec, default=0.5,
                      help="cluster virial radius [%default]")
    result.add_option("-Q", dest="Qvir", type="float", default=0.5,
                      help="virial ratio [%default]")

    # Disk parameters
    result.add_option("-a", dest="alpha", type="float", default=1E-4,
                      help="turbulence parameter [%default]")
    result.add_option("-c", dest="R", type="float", default=30.0,
                      help="Initial disk radius [%default]")

    result.add_option("-e", dest="gas_expulsion_onset", type="float", default=0.6 | units.Myr,
                      help="the moment when the gas starts dispersing [%default]")
    result.add_option("-E", dest="gas_expulsion_timescale", type="float", default=0.1 | units.Myr,
                      help="the time after which half of the initial gas is expulsed assuming gas Plummer radius of 1 parsec [%default]")

    # Time parameters
    result.add_option("-I", dest="t_ini", type="int", default=0 | units.yr,
                      help="initial time [%default]")
    result.add_option("-t", dest="dt", type="int", default=2000 | units.yr,
                      help="time interval of recomputing circumstellar disk sizes and checking for energy conservation [%default]")
    result.add_option("-x", dest="t_end", type="float", default=2 | units.Myr,
                      help="end time of the simulation [%default]")

    # Gas behaviour
    result.add_option("-l", dest="gas_presence", action="store_false", default=False,
                      help="gas presence [%default]")
    result.add_option("-k", dest="gas_expulsion", action="store_false", default=False,
                      help="gas expulsion [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)