from amuse.lab import *
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
import numpy
from matplotlib import pyplot


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
    stellar_evolution.particles.add_particles(Particles(mass=star.mass))
    stellar_luminosity = stellar_evolution.particles.luminosity.value_in(units.LSun)
    stellar_evolution.stop()

    R = star.initial_characteristic_disk_radius
    T = Tref * (stellar_luminosity ** 0.25)
    q = temperature_profile
    M = constants.G * star.mass

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
    #print(grid)

    t_end = t_end | units.Myr

    # Test run: 2 disks + one massive star
    max_stellar_mass = 2 | units.MSun
    stellar_masses = new_kroupa_mass_distribution(3, max_stellar_mass)  # , random=False)
    disk_masses = 0.1 * stellar_masses
    converter = nbody_system.nbody_to_si(stellar_masses.sum() + disk_masses.sum(), Rvir)

    stars = new_plummer_model(N, converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    print("stellar masses: ", stellar_masses)

    stars.mass = stellar_masses
    stars.initial_characteristic_disk_radius = (stars.mass.value_in(units.MSun) ** 0.5) * R | units.AU
    stars.disk_radius = stars.initial_characteristic_disk_radius
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

    #print("star.mass: ", stars.mass)
    #print("star.mass: ", stars.mass.value_in(units.MSun))

    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    #stellar.particles.add_particles(Particles(mass=stars.stellar_mass))
    stellar.particles.add_particles(stars)

    print("star.mass: ", stars.mass.value_in(units.MSun))

    initial_luminosity = stellar.particles.luminosity
    stars.luminosity = initial_luminosity
    stars.temperature = stellar.particles.temperature
    dt = 5 | units.Myr

    print("L(t=0 Myr) = {0}".format(initial_luminosity))
    print("R(t=0 Myr) = {0}".format(stellar.particles.radius.in_(units.RSun)))
    print("m(t=0 Myr) = {0}".format(stellar.particles.mass.in_(units.MSun)))

    channel_to_framework = stellar.particles.new_channel_to(stars)
    write_set_to_file(stars, 'results/0.hdf5', 'amuse')

    while stellar.model_time < t_end:
        stellar.evolve_model(stellar.model_time + dt)
        channel_to_framework.copy_attributes(["radius", "temperature",
                                              "luminosity"])
        write_set_to_file(stars, 'results/{0}.hdf5'.format(int(stellar.model_time.value_in(units.Myr))), 'amuse')

    #stellar.evolve_model(t_end)

    print("***")
    print("L(t={0}) = {1}".format(stellar.model_time, stellar.particles.luminosity.in_(units.LSun)))
    print("R(t={0}) = {1}".format(stellar.model_time, stellar.particles.radius.in_(units.RSun)))
    print("m(t={0}) = {1}".format(stellar.model_time, stellar.particles.mass.in_(units.MSun)))

    stellar.stop()

    times = numpy.arange(0, 10005, 5)
    T1, L1, R1 = [], [], []
    T2, L2, R2 = [], [], []
    T3, L3, R3 = [], [], []

    for t in times:
        stars = read_set_from_file('results/{0}.hdf5'.format(t), 'hdf5')
        T1.append(stars[0].temperature.value_in(units.K))
        L1.append(stars[0].luminosity.value_in(units.LSun))
        R1.append(stars[0].radius.value_in(units.RSun))

        T2.append(stars[1].temperature.value_in(units.K))
        L2.append(stars[1].luminosity.value_in(units.LSun))
        R2.append(stars[1].radius.value_in(units.RSun))

        T3.append(stars[2].temperature.value_in(units.K))
        L3.append(stars[2].luminosity.value_in(units.LSun))
        R3.append(stars[2].radius.value_in(units.RSun))

    x_label = "T [K]"
    y_label = "L [L$_\odot$]"
    fig = pyplot.figure(figsize=(10, 8), dpi=90)
    ax = pyplot.subplot(111)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    ax.set_yscale('log')
    pyplot.gca().invert_xaxis()
    pyplot.scatter(T1, L1, s=80*numpy.sqrt(R1), c=times, cmap='Greens')
    pyplot.scatter(T2, L2, s=80*numpy.sqrt(R2), c=times, cmap='Blues')
    pyplot.scatter(T3, L3, s=80*numpy.sqrt(R3), c=times, cmap='Oranges')

    pyplot.plot(T1[0], L1[0], 'rx')
    pyplot.plot(T2[0], L2[0], 'rx')
    pyplot.plot(T3[0], L3[0], 'rx')

    pyplot.show()

    """stars = read_set_from_file('stars.h5', 'hdf5')
    T = stars.temperature.value_in(units.K)
    L = stars.luminosity.value_in(units.LSun)
    R = stars.radius.value_in(units.RSun)

    #R = 80 * numpy.sqrt(R)
    pyplot.scatter(T, L, c='b', lw=0)
    pyplot.show()"""


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