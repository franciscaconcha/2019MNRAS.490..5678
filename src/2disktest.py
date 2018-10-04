from amuse.lab import *
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
import numpy
from matplotlib import pyplot
import gzip
import copy


def luminosity_fit(mass):
    """
    Return stellar luminosity (in LSun) for corresponding mass, as calculated with Martijn's fit

    :param mass: stellar mass in MSun
    :return: stellar luminosity in LSun
    """
    if 0.12 < mass < 0.24:
        return (1.70294E16 * numpy.power(mass, 42.557)) | units.LSun
    elif 0.24 < mass < 0.56:
        return (9.11137E-9 * numpy.power(mass, 3.8845)) | units.LSun
    elif 0.56 < mass < 0.70:
        return (1.10021E-6 * numpy.power(mass, 12.237)) | units.LSun
    elif 0.70 < mass < 0.91:
        return (2.38690E-4 * numpy.power(mass, 27.199)) | units.LSun
    elif 0.91 < mass < 1.37:
        return (1.02477E-4 * numpy.power(mass, 18.465)) | units.LSun
    elif 1.37 < mass < 2.07:
        return (9.66362E-4 * numpy.power(mass, 11.410)) | units.LSun
    elif 2.07 < mass < 3.72:
        return (6.49335E-2 * numpy.power(mass, 5.6147)) | units.LSun
    elif 3.72 < mass < 10.0:
        return (6.99075E-1 * numpy.power(mass, 3.8058)) | units.LSun
    elif 10.0 < mass < 20.2:
        return (9.73664E0 * numpy.power(mass, 2.6620)) | units.LSun
    elif 20.2 < mass:
        return (1.31175E2 * numpy.power(mass, 1.7974)) | units.LSun
    else:
        return 0 | units.LSun


def find_nearest(array, value):
    """
    Return closest number to "value" in array
    :param array: Array of floats
    :param value: Value to find
    :return: Index of most similar number to value, and the number
    """
    array = numpy.asarray(array)
    idx = numpy.abs(array - value).argmin()
    return idx, array[idx]


def read_UVBLUE(filename, limits=None):
    """
    Read UVBLUE spectrum

    :param filename: Name of file to read, including path
    :param limits: [low, high] Lower and higher wavelength limits to read. If not specified, the whole spectrum is returned
    :return: Array with radiation in the range given by limits, or full wavelength range
    """

    column1 = []  # data
    column2 = []  # "fit"?

    with gzip.open(filename, 'r') as fuv:
        for line in fuv.readlines()[3:]:
            l1, l2 = line.split()
            column1.append(float(l1))
            column2.append(float(l2))

    if limits is not None:
        with gzip.open(filename, 'r') as fuv:
            wl = fuv.readlines()[1].split()
        steps, first, last = int(wl[0]), float(wl[1]), float(wl[2])

        # Find the correct range in the wavelengths, return corresponding radiation range
        wavelengths = numpy.linspace(first, last, steps)

        id_lower, lower = find_nearest(wavelengths, limits[0])
        id_higher, higher = find_nearest(wavelengths, limits[1])

        FUV_radiation = column1[id_lower:id_higher + 1]  # + 1 to include higher
        return numpy.array(FUV_radiation) | units.erg
    else:
        return numpy.array(column1)

    #pyplot.plot(column1)
    #pyplot.plot(column2)
    #pyplot.show()
    #return numpy.array(column1)


def integrate_FUV(filename, lower, higher):
    """
    Return total FUV radiation in SED between wavelenghts lower and higher.
    :param filename: Name for UVBLUE file to read
    :param lower: Lower limit for wavelength
    :param higher: Higher limit for wavelength
    :return: Total radiation between lower and higher wavelengths
    """
    radiation = read_UVBLUE(filename, [lower, higher])
    return radiation.sum()


def distance(star1, star2):
    """
    Return distance between star1 and star2
    :param star1:
    :param star2:
    :return:
    """
    return numpy.sqrt((star2.x - star1.x)**2 + (star2.y - star1.y)**2 + (star2.z - star1.z)**2)


def radiation_at_distance(rad, R):
    """
    Return radiation rad at distance R
    :param rad: total radiation of star in erg/s
    :param R: distance in cm
    :return: radiation of star at distance R, in erg * s^-1 * cm^-2
    """
    return rad / (4 * numpy.pi * R**2) | (units.erg / (units.s * units.cm**2))


def find_indices(column, val):
    """
    Return indices of column values in between which val is located.
    Return i,j such that column[i] < val < column[j]

    :param column: column where val is to be located
    :param val: number to be located in column
    :return: i, j indices
    """
    """index_i = min(range(len(column)), key=lambda i: abs(column[i] - val))
    index_i_value = column[index_i]

    if index_i_value <= val:
        index_j = 0
        N = index_i

        for i in column[index_i:]:
            if i > index_i_value:
                index_j = N
                break
            else:
                N += 1
    else:
        index_j = 0
        N = index_i

        for i in column[:index_i]:
            if i > index_i_value:
                index_j = N
                break
            else:
                N += 1

    #if index == len(column) - 1: # If most similar value is found at the end of the column
    #    return index, index

    return index_i, index_j"""
    # the largest element of myArr less than myNumber
    value_below = column[column < val].max()
    index_i = numpy.where(column == value_below)[0][0]

    # the smallest element of myArr greater than myNumber
    value_above = column[column > val].min()
    index_j = numpy.where(column == value_above)[0][0]

    return index_i, index_j



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
    print("Temp(t=0 Myr) = {0}".format(stellar.particles[2].temperature.in_(units.K)))

    channel_to_framework = stellar.particles.new_channel_to(stars)
    write_set_to_file(stars, 'results/0.hdf5', 'amuse')

    #temp, temp2 = [], []
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
    grid_disk_radius = FRIED_grid[:, 3]

    # Stellar masses of the FRIED grid (MSun)
    #masses = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 1.9]
    #FUV_fields = [10, 100, 1000, 5000, 10000]
    #for m in FUV_fields:
    #    M = [x for x in grid if x[1] == m]
    #    print len(M)
    #    print M[0][0]

    while time < t_end:
        dt = min(dt, t_end - time)
        stellar.evolve_model(time + dt/2)
        channel_from_stellar_to_gravity.copy()
        Etot_gr = gravity.kinetic_energy + gravity.potential_energy
        gravity.evolve_model(time + dt)
        dE_gr += (gravity.kinetic_energy + gravity.potential_energy - Etot_gr)

        #channel_to_framework.copy_attributes(["radius", "temperature",
        #                                      "luminosity"])
        #temp.append(stellar.particles[2].temperature.value_in(units.K))
        #temp2.append(round(stellar.particles[2].temperature.value_in(units.K) / 500) * 500)

        for s in bright_stars:  # For each massive/bright star
            temp = round(s.temperature.value_in(units.K) / 500) * 500
            #print("temp: {0}".format(temp))

            if temp < 3500:
                g = "50"
            elif temp > 24000:
                g = "40"
            else:
                g = "45"

            temp_5d = format(int(temp), "05d")  # Correct format for UVBLUE filename
            #rad = integrate_FUV(fuv_filename_base.format(temp_5d, g), 1000, 3000)  # 1000, 3000 A: FUV limits
            lum = luminosity_fit(s.mass.value_in(units.MSun))  # In LSun

            for ss in small_stars:
                dist = distance(s, ss)
                radiation_ss = radiation_at_distance(lum.value_in(units.erg / units.s),
                                                     dist.value_in(units.cm))
                radiation_ss_G0 = radiation_ss.value_in(units.erg/(units.s * units.cm**2)) / 1.6E-3
                #print("Radiation in star {0} at time {1}: {2} G0".format(list(stellar.particles).index(ss),
                #                                                         stellar.model_time,
                #                                                        radiation_ss.value_in(units.erg/(units.s * units.cm**2)) / 1.6E-3))
                print(ss.mass.value_in(units.MSun),
                      radiation_ss_G0,
                      ss.disk_mass.value_in(units.MJupiter),
                      ss.disk_radius.value_in(units.AU)
                      )
                subgrid = numpy.ndarray(shape=(8, 4), dtype=float, order='F')
                #print("Mass indices:")
                stellar_mass_i, stellar_mass_j = find_indices(grid_stellar_masses, ss.mass.value_in(units.MSun))
                subgrid[0] = FRIED_grid[stellar_mass_i]
                subgrid[1] = FRIED_grid[stellar_mass_j]
                print stellar_mass_i, stellar_mass_j
                print grid_stellar_masses[stellar_mass_i], grid_stellar_masses[stellar_mass_j]
                #print("FUV indices:")
                FUV_i, FUV_j =  find_indices(grid_FUV, radiation_ss_G0)
                subgrid[2] = FRIED_grid[FUV_i]
                subgrid[3] = FRIED_grid[FUV_j]
                print FUV_i, FUV_j
                print grid_FUV[FUV_i], grid_FUV[FUV_j]
                #print("Disk mass indices:")
                disk_mass_i, disk_mass_j = find_indices(grid_disk_mass, ss.disk_mass.value_in(units.MJupiter))
                subgrid[4] = FRIED_grid[disk_mass_i]
                subgrid[5] = FRIED_grid[disk_mass_j]
                print disk_mass_i, disk_mass_j
                print grid_disk_mass[disk_mass_i], grid_disk_mass[disk_mass_j]
                #print("Disk radius indices:")
                disk_radius_i, disk_radius_j = find_indices(grid_disk_radius, ss.disk_radius.value_in(units.AU))
                subgrid[6] = FRIED_grid[disk_radius_i]
                subgrid[7] = FRIED_grid[disk_radius_j]
                print disk_radius_i, disk_radius_j
                print grid_disk_radius[disk_radius_i], grid_disk_radius[disk_radius_j]
                print subgrid
                break
            break
        break

        stellar.evolve_model(time + dt)
        channel_from_stellar_to_gravity.copy()
        channel_from_gravity_to_framework.copy()
        time += dt
        #print(time)

    print "END:"
    print stars.x
    print stars.y
    print stars.z
    pyplot.scatter(initx.value_in(units.parsec), inity.value_in(units.parsec), s=500, label="init")
    pyplot.scatter(stars.x.value_in(units.parsec), stars.y.value_in(units.parsec),
                   s=100 * stars[2].radius.value_in(units.RSun), label="end")
    pyplot.legend()
    pyplot.show()

        #print(round(stellar.particles[2].temperature.value_in(units.K) / 500) * 500)
        #write_set_to_file(stars, 'results/{0}.hdf5'.format(int(stellar.model_time.value_in(units.Myr))), 'amuse')

    #stellar.evolve_model(t_end)

    print("***")
    print("L(t={0}) = {1}".format(stellar.model_time, stellar.particles.luminosity.in_(units.LSun)))
    print("R(t={0}) = {1}".format(stellar.model_time, stellar.particles.radius.in_(units.RSun)))
    print("m(t={0}) = {1}".format(stellar.model_time, stellar.particles.mass.in_(units.MSun)))
    print("Temp(t={0}) = {1}".format(stellar.model_time, stellar.particles[2].temperature.in_(units.K)))


    stellar.stop()

    #pyplot.plot(temp, label="Original temperatures")
    #pyplot.plot(temp2, label="Temperatures rounded to closest 500")
    #pyplot.legend()
    #pyplot.show()

    """times = numpy.arange(0, 10005, 5)
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
    #pyplot.gca().invert_xaxis()
    #pyplot.scatter(T1, L1, s=80*numpy.sqrt(R1), c=times, cmap='Greens')
    #pyplot.scatter(T2, L2, s=80*numpy.sqrt(R2), c=times, cmap='Blues')
    #pyplot.scatter(T3, L3, s=80*numpy.sqrt(R3), c=times, cmap='Oranges')

    pyplot.plot(times, L1)
    pyplot.plot(times, L2)
    pyplot.plot(times, L3)

    #pyplot.plot(T1[0], L1[0], 'rx')
    #pyplot.plot(T2[0], L2[0], 'rx')
    #pyplot.plot(T3[0], L3[0], 'rx')

    pyplot.show()"""

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