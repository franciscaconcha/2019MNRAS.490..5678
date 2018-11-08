from amuse.lab import *
import numpy
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from matplotlib import pyplot
import gzip
import copy
from scipy import interpolate
import Queue
import threading
import multiprocessing
import sys
from decorators import timer


code_queue = Queue.Queue()


def column_density(grid, r0, mass, lower_density=1E-12 | units.g / units.cm**2):
    r = grid.value_in(units.AU) | units.AU
    rd = r0
    Md = mass

    Sigma_0 = Md / (2 * numpy.pi * r0 ** 2 * (1 - numpy.exp(-rd / r0)))
    Sigma = Sigma_0 * (r0 / r) * numpy.exp(-r / r0) * (r <= r0) + lower_density
    return Sigma


def initialize_vader_code(disk_radius, disk_mass, r_min=0.5 | units.AU, r_max=5000 | units.AU, n_cells=500, linear=True):
    """ Initialize vader code for given parameters.

    :param disk_radius: disk radius. Must have units.Au
    :param disk_mass: disk mass. Must have units.MSun
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

    return disk


def remote_worker_code(dt):
    code = code_queue.get()
    evolve_single_disk(code, dt)
    code_queue.task_done()


def evolve_parallel_disks(codes, dt):
    for ci in codes:
        code_queue.put(ci)
    n_cpu = multiprocessing.cpu_count()
    for i in range(n_cpu):
        th = threading.Thread(target=remote_worker_code, args=[dt])
        th.daemon = True
        th.start()
    code_queue.join() 	    # block until all tasks are done


def evolve_single_disk(code, time):
    disk = code
    disk.evolve_model(time)
    disk.stop()


def distance(star1, star2):
    """ Return distance between star1 and star2

    :param star1:
    :param star2:
    :return:
    """
    return numpy.sqrt((star2.x - star1.x)**2 + (star2.y - star1.y)**2 + (star2.z - star1.z)**2)


def radiation_at_distance(rad, R):
    """ Return radiation rad at distance R

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

    # The largest element of column less than val
    try:
        value_below = column[column < val].max()
    except ValueError:
        # If there are no values less than val in column, return smallest element of column
        value_below = column.min()
    # Find index
    index_i = numpy.where(column == value_below)[0][0]

    # The smallest element of column greater than val
    try:
        value_above = column[column > val].min()
    except ValueError:
        # If there are no values larger than val in column, return largest element of column
        value_above = column.max()
    # Find index
    index_j = numpy.where(column == value_above)[0][0]

    return int(index_i), int(index_j)


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


def periastron_distance(stars):
    """ Return the periastron distance of two encountering stars.
    :param stars: pair of encountering stars.
    :return: periastron distance of the encounter.
    """
    # Standard gravitational parameter
    mu = constants.G * stars.mass.sum()

    # Position vector from one star to the other
    r = stars[0].position - stars[1].position

    # Relative velocity between the stars
    v = stars[0].velocity - stars[1].velocity

    # Energy
    E = (v.length()) ** 2 / 2 - mu / r.length()

    # Semi-major axis
    a = -mu / 2 / E

    # Semi-latus rectum
    p = (numpy.cross(r.value_in(units.AU),
                  v.value_in(units.m / units.s)) | units.AU * units.m / units.s).length() ** 2 / mu

    # Eccentricity
    e = numpy.sqrt(1 - p / a)

    # Periastron distance
    return p / (1 + e)


def get_disk_radius(disk, density_limit=1E-12):
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


def resolve_encounter(stars,
                      disk_codes,
                      time,
                      mass_factor_exponent,
                      truncation_parameter,
                      gamma,
                      verbose=False):
    """ Resolve encounter between two stars.
        Return new disk radii and masses to create new vader codes.

    :param stars: pair of encountering stars.
    :param disk_codes: vader codes of the disks in the encounter.
    :param time: time at which encounter occurs.
    :param mass_factor_exponent: exponent characterizing truncation mass dependence in a stellar encounter (eq. 13).
    :param truncation_parameter: factor characterizing the size of circumstellar disks after an encounter (eq. 13).
    :param gamma: radial viscosity dependence exponent.
    :param verbose: verbose option for debugging.
    """
    # For debugging
    if verbose:
        print(time.value_in(units.yr), stars.mass.value_in(units.MSun))

    closest_approach = periastron_distance(stars)

    new_radii, new_masses = [], []

    # Check each star
    for i in range(2):
        truncation_radius = (closest_approach.value_in(units.AU) / 3) *\
                            numpy.sqrt(stars[i].stellar_mass.value_in(units.MSun)
                                       / stars[1 - i].stellar_mass.value_in(units.MSun)) | units.AU
        #truncation_radius = (closest_approach * truncation_parameter * \
        #                    ((stars[i].stellar_mass / stars[1 - i].stellar_mass) ** mass_factor_exponent)).value_in(units.AU) | units.AU

        if stars[i].disk_radius == 0 | units.AU:
            new_radii.append(None)
            continue

        R_disk = get_disk_radius(disk_codes[i])
        print "R_disk = {0}, truncation radius={1}".format(R_disk, truncation_radius)

        # Update collisional radius so that we don't detect this encounter in the next time step
        stars[i].collisional_radius = 0.49 * closest_approach

        if truncation_radius < R_disk:
            print "truncating encounter"
            # Add accreted mass to star. Not sure if this should go here.
            #stars[i].stellar_mass += stars[i].initial_disk_mass - get_disk_mass(stars[i], time, gamma)
            stars[i].last_encounter = time

            print "star radius: {0} truncation radius: {1}".format(stars[i].disk_radius, truncation_radius)
            new_radii.append(truncation_radius)
            new_masses.append(get_disk_mass(disk_codes[i], truncation_radius))
        else:
            "previous values..."
            new_radii.append(R_disk)
            new_masses.append(stars[i].disk_mass)

    return new_radii, new_masses


@timer
def main(N, Rvir, Qvir, alpha, R, t_ini, t_end, save_interval, run_number, save_path,
         gamma=1,
         mass_factor_exponent=0.2,
         truncation_parameter=1. / 3,
         plummer_radius=0.5 | units.parsec,
         dt=2000 | units.yr,
         temp_profile=0.5,
         Rref=1.0 | units.AU,
         Tref=280 | units.K,
         mu=2.3 | units.g / units.mol,
         filename=''):

    try:
        float(t_end)
        t_end = t_end | units.Myr
    except TypeError:
        pass

    max_stellar_mass = 100 | units.MSun
    stellar_masses = new_kroupa_mass_distribution(N, max_stellar_mass)  # , random=False)
    converter = nbody_system.nbody_to_si(stellar_masses.sum(), Rvir)
    stars = new_plummer_model(N, converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    stars.stellar_mass = stellar_masses

    # Bright stars: no disks; emit FUV radiation
    bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
    if len(bright_stars) == 0:
        stars[0].stellar_mass = 3 | units.MSun
        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        print("Warning: No star with mass > 1.9 MSun generated by the IMF."
              "\nOne star of 3 MSun added to the simulation.")

    # Small stars: with disks; radiation not considered
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) < 1.9]

    small_stars.disk_radius = 100 * (small_stars.stellar_mass.value_in(units.MSun) ** 0.5) | units.AU
    bright_stars.disk_radius = 100 | units.AU

    bright_stars.disk_mass = 0 | units.MSun
    small_stars.disk_mass = 0.1 * small_stars.stellar_mass

    # Initially all stars have the same collisional radius
    # TODO if we get too few encounters, we can make this smaller
    stars.collisional_radius = 0.02 | units.parsec

    disk_codes = []
    disk_codes_indices = {}  # Using this to keep track of codes later on, for the encounters

    # Create individual instances of vader codes for each disk
    for s in small_stars:
        s_code = initialize_vader_code(s.disk_radius, s.disk_mass, linear=False)

        s_code.parameters.alpha = alpha
        s_code.parameters.verbosity = False
        s_code.parameters.post_timestep_function = True
        s_code.parameters.maximum_tolerated_change = 1E99
        s_code.parameters.number_of_user_parameters = 6
        s_code.parameters.inner_pressure_boundary_torque = 0. | units.g * units.cm**2. / units.s**2.
        s_code.set_parameter(0, 0.)
        s_code.set_parameter(2, 1E-12)
        s_code.set_parameter(3, 300)
        s_code.set_parameter(4, 2.33 * constants.u.value_in(units.g) * 1.008)

        disk_codes.append(s_code)
        disk_codes_indices[s.key] = len(disk_codes) - 1

    print small_stars[0].disk_radius
    print get_disk_radius(disk_codes[0], density_limit=1E-11)
    print disk_codes[0]
    print disk_codes[0].grid
    print disk_codes[0].grid.column_density
    disk_codes[0].evolve_model(0.02 | units.Myr)
    print get_disk_radius(disk_codes[0], density_limit=1E-11)
    print disk_codes[0].grid.column_density

    # Start gravity code, add all stars
    gravity = ph4(converter)
    gravity.parameters.timestep_parameter = 0.01
    gravity.parameters.epsilon_squared = (100 | units.AU) ** 2
    gravity.particles.add_particles(stars)

    # Enable stopping condition for dynamical encounters
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

    # Start stellar evolution code, add only massive stars
    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    stellar.particles.add_particles(bright_stars)

    # Communication channels
    channel_from_stellar_to_framework = stellar.particles.new_channel_to(stars)
    channel_from_stellar_to_gravity = stellar.particles.new_channel_to(gravity.particles)
    channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars)
    channel_from_framework_to_gravity = stars.new_channel_to(gravity.particles,
                                                             attributes=['collisional_radius'],
                                                             target_names=['radius'])

    ######## FRIED grid ########
    # Read FRIED grid
    grid = numpy.loadtxt('friedgrid.dat', skiprows=2)

    # Getting only the useful parameters from the grid (not including Mdot)
    FRIED_grid = grid[:, [0, 1, 2, 4]]
    grid_log10Mdot = grid[:, 5]

    grid_stellar_mass = FRIED_grid[:, 0]
    grid_FUV = FRIED_grid[:, 1]
    grid_disk_mass = FRIED_grid[:, 2]
    grid_disk_radius = FRIED_grid[:, 3]

    t = 0 | t_end.unit
    E_ini = gravity.kinetic_energy + gravity.potential_energy

    # For keeping track of energy
    #E_handle = file('{0}/energy.txt'.format(path), 'a')
    #Q_handle = file('{0}/virial.txt'.format(path), 'a')
    E_list = []
    Q_list = []

    print stellar.particles.luminosity

    print small_stars.disk_radius

    # Evolve!
    while t < t_end:
        dt = min(dt, t_end - t)
        stellar.evolve_model(t + dt/2)
        channel_from_stellar_to_gravity.copy()
        E_kin = gravity.kinetic_energy
        E_pot = gravity.potential_energy

        E_list.append([(E_kin + E_pot) / E_ini - 1])
        Q_list.append([-1.0 * E_kin / E_pot])

        # Update the collision radii of the stars based on the truncation factors and viscous spreading.
        # TODO update the collisional radius...
        gravity.particles.radius = stars.collisional_radius

        gravity.evolve_model(t + dt)

        if stopping_condition.is_set():
            print("encounter")
            channel_from_gravity_to_framework.copy()
            encountering_stars = Particles(particles=[stopping_condition.particles(0)[0],
                                                      stopping_condition.particles(1)[0]])

            code_index = [disk_codes_indices[encountering_stars[0].key], disk_codes_indices[encountering_stars[1].key]]

            #print disk_codes[code_index[0]].grid

            new_radii, new_masses = resolve_encounter(encountering_stars.get_intersecting_subset_in(stars),
                                                      [disk_codes[code_index[0]],
                                                       disk_codes[code_index[1]]],
                                                      gravity.model_time + t_ini,
                                                      mass_factor_exponent,
                                                      truncation_parameter,
                                                      gamma)
            #print encountering_stars.get_intersecting_subset_in(stars).disk_radius
            print new_radii

            if new_radii[0] is not None and new_radii[1] is not None:
                print "small small"
                # If encounter did not truncate, new_radii and new_masses will be empty
                # Going to take the old codes involved in the encounter and replace them with new codes
                #print("new radii!")
                for i in range(2):
                    if new_radii[i] == encountering_stars.get_intersecting_subset_in(stars)[i].disk_radius:
                        continue
                    else:
                        disk = disk_codes[code_index[i]]
                        sigma = column_density(disk.grid.r, new_radii[i], new_masses[i])
                        disk_codes[code_index[i]].grid.column_density = sigma
            elif new_radii[0] is None:
                print "big"
            else:
                print "big"

        # Copy stars' new collisional radii (updated in resolve_encounter) to gravity
        channel_from_framework_to_gravity.copy()
        stellar.evolve_model(t + dt)
        channel_from_stellar_to_gravity.copy()
        channel_from_stellar_to_framework.copy()

        """"print "before evolving"
        print disk_codes[0].grid.r
        evolve_parallel_disks(disk_codes, t + dt)
        #evolve_single_disk(disk_codes[0], t+dt)
        #disk_codes[0].evolve_model(t + dt)
        print "after evolving"
        print disk_codes[0].grid.r"""

        """print "going to run disk codes..."
        for s in small_stars:
            disk_codes[disk_codes_indices[s.key]].evolve_model(t + dt)
            s.disk_radius = get_disk_radius(disk_codes[disk_codes_indices[s.key]], density_limit=1E-6)
        print "done"""""

        t += dt

    """if (t + t_ini).value_in(units.yr) % save_interval.value_in(units.yr) == 0:
            channel_from_gravity_to_framework.copy()

            np.savetxt(E_handle, E_list)
            np.savetxt(Q_handle, Q_list)

            E_list = []
            Q_list = []

            # TODO this was done every few timesteps to calculate viscous growth. Now here I should do photoevap!
            # and I think it should go on every time step, not in intervals

            new_mass = get_disk_mass(stars, t + t_ini, gamma)
            new_radius = disk_characteristic_radius(stars, t + t_ini, gamma)

            stars.stellar_mass += stars.initial_disk_mass - new_mass
            stars.initial_disk_mass = new_mass
            stars.viscous_timescale *= (np.divide(new_radius, stars.initial_characteristic_disk_radius)) ** (2 - gamma)
            stars.initial_characteristic_disk_radius = new_radius
            stars.last_encounter = t + t_ini

            write_set_to_file(stars,
                              '{0}/R{1}_{2}.hdf5'.format(path, Rvir.value_in(units.parsec),
                                                         int((t + t_ini).value_in(units.yr))),
                              'amuse')"""

    print stellar.particles.luminosity
    print small_stars.disk_radius
    gravity.stop()
    #E_handle.close()
    #Q_handle.close()
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
    result.add_option("-a", dest="alpha", type="float", default=1E-4,
                      help="turbulence parameter [%default]")
    result.add_option("-c", dest="R", type="float", default=30.0,
                      help="Initial disk radius [%default]")

    # Time parameters
    result.add_option("-I", dest="t_ini", type="int", default=0 | units.yr,
                      help="initial time [%default]")
    result.add_option("-t", dest="dt", type="int", default=2000 | units.yr,
                      help="time interval of recomputing circumstellar disk sizes and checking for energy conservation [%default]")
    result.add_option("-e", dest="t_end", type="float", default=2 | units.Myr,
                      help="end time of the simulation [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

