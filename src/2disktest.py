from amuse.lab import *
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
import numpy


def viscous_timescale(star, alpha, temperature_profile, Rref, Tref, mu, gamma):
    """Compute the viscous timescale of the circumstellar disk.

    :param star: star with the circumstellar disk.
    :param alpha: turbulence mixing strenght.
    :param temperature_profile: negative of the temperature profile exponent, q in eq. (8).
    :param Rref: reference distance from the star at which the disk temperature is given.
    :param Tref: disk temperature at the reference distance for a star with solar luminosity.
    :param mu: molar mass of the gas in g/mol.
    :param gamma: radial viscosity dependence exponent.
    :return: viscous timescale in Myr
    """
    # To calculate luminosity
    stellar_evolution = SeBa()
    stellar_evolution.particles.add_particles(Particles(mass=star.stellar_mass))
    stellar_luminosity = stellar_evolution.particles.luminosity.value_in(units.LSun)
    stellar_evolution.stop()

    R = star.initial_characteristic_disk_radius
    T = Tref * (stellar_luminosity ** 0.25)
    q = temperature_profile
    M = constants.G * star.stellar_mass

    return mu * (R ** (0.5 + q)) * (M ** 0.5) / 3 / alpha / ((2 - gamma) ** 2) \
           / constants.molar_gas_constant / T / (Rref ** q)


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

    # Read FRIED grid
    grid = numpy.loadtxt('friedgrid.dat', skiprows=2)
    print(grid)

    t_end = t_end | units.Myr

    # Test run: 2 disks + one massive star
    max_stellar_mass = 2 | units.MSun
    stellar_masses = new_kroupa_mass_distribution(3, max_stellar_mass)  # , random=False)
    disk_masses = 0.1 * stellar_masses
    converter = nbody_system.nbody_to_si(stellar_masses.sum() + disk_masses.sum(), Rvir)

    stars = new_plummer_model(N, converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    print(stellar_masses)

    stars.stellar_mass = stellar_masses
    stars.initial_characteristic_disk_radius = (stars.stellar_mass.value_in(units.MSun) ** 0.5) * R | units.AU
    stars.disk_radius = stars.initial_characteristic_disk_radius
    stars.initial_disk_mass = disk_masses
    stars.disk_mass = stars.initial_disk_mass
    stars.total_star_mass = stars.initial_disk_mass + stars.stellar_mass
    stars.viscous_timescale = viscous_timescale(stars, alpha, temp_profile, Rref, Tref, mu, gamma)
    stars.last_encounter = 0.0 | units.yr

    # Bright star
    stars[2].stellar_mass = 5 | units.MSun
    stars[2].initial_characteristic_disk_radius = 0 | units.AU
    stars[2].initial_disk_mass = 0 | units.MSun
    stars[2].total_star_mass = stars[2].stellar_mass
    stars[2].viscous_timescale = 0 | units.yr

    print stars.total_star_mass.value_in(units.MSun)
    print stars.stellar_mass.value_in(units.MSun)

    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    stellar.particles.add_particles(Particles(mass=stars.stellar_mass))

    initial_luminosity = stellar.particles.luminosity
    dt = 5 | units.Myr

    print("L(t=0 Myr) = {0}".format(initial_luminosity))
    print("R(t=0 Myr) = {0}".format(stellar.particles.radius.in_(units.RSun)))
    print("m(t=0 Myr) = {0}".format(stellar.particles.mass.in_(units.MSun)))

    while stellar.model_time < t_end:
        stellar.evolve_model(stellar.model_time + dt)

    print("***")
    print("L(t={0}) = {1}".format(stellar.model_time, stellar.particles.luminosity.in_(units.LSun)))
    print("R(t={0}) = {1}".format(stellar.model_time, stellar.particles.radius.in_(units.RSun)))
    print("m(t={0}) = {1}".format(stellar.model_time, stellar.particles.mass.in_(units.MSun)))

    stellar.stop()


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
    result.add_option("-a", dest="alpha", type="float", default=1E-2,
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