from amuse.lab import *
import numpy
from random import randint
from amuse import io
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
import time
import os

# Workaround for now
global diverged_disks, disk_codes_indices
diverged_disks = {}
disk_codes_indices = {}


def column_density(grid, r0, mass, lower_density=1E-12 | units.g / units.cm**2):
    r = grid.value_in(units.AU) | units.AU
    rd = r0
    Md = mass

    Sigma_0 = Md / (2 * numpy.pi * r0 ** 2 * (1 - numpy.exp(-rd / r0)))
    Sigma = Sigma_0 * (r0 / r) * numpy.exp(-r / r0) * (r <= r0) + lower_density
    return Sigma


def initialize_vader_code(disk_radius, disk_mass, alpha, r_min=0.05 | units.AU, r_max=2000 | units.AU, n_cells=100, linear=True):
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
    global diverged_disks
    diverged_disks[disk] = False
    #disk.set_parameter(0, False)  # Disk parameter for non-convergence. True: disk diverged

    

    return disk


def evolve_parallel_disks(codes, dt):
    n_cpu = multiprocessing.cpu_count()
    processes = []
    threads = []

    #print "Starting processes... n_cpu = {0}".format(n_cpu)

    for i in range(len(codes)):
        p = multiprocessing.Process(name=str(i), target=evolve_single_disk, args=(codes[i], dt, ))
        processes.append(p)
        p.start()
        #th = threading.Thread(target=evolve_single_disk, args=[codes[i], dt])
        #th.daemon = True
        #threads.append(th)
        #th.start()

    #for t in threads:
    #    t.join()

    for p in processes:
        p.join()

    #print "All processes finished"


def evolve_single_disk(code, dt):
    #print "current process: {0}".format(multiprocessing.current_process().name)
    disk = code
    try:
        disk.evolve_model(dt)
    except:
        print "Disk did not converge"
        global diverged_disks
        diverged_disks[disk] = True
        #disk.parameters.inner_pressure_boundary_type = 3
        #disk.parameters.inner_boundary_function = False


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


def get_disk_radius(disk, f=0.95):

    Mtot = (disk.grid.area * disk.grid.column_density).sum()
    Mcum = 0. | units.MSun

    edge = -1

    for i in range(len(disk.grid.r)):
        Mcum += disk.grid.area[i] * disk.grid.column_density[i]

        if Mcum >= Mtot * f:
            edge = i
            break

    return disk.grid.r[edge].value_in(units.au) | units.au


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


def accretion_rate(mass):

    return numpy.power(10, (1.89 * numpy.log10(mass.value_in(units.MSun)) - 8.35)) | units.MSun / units.yr


def resolve_encounter(stars,
                      disk_codes,
                      time,
                      verbose=False):
    """ Resolve encounter between two stars.
        Return updated vader codes.

    :param stars: pair of encountering stars
    :param disk_codes: vader codes of the disks in the encounter
    :param time: time at which encounter occurs
    :param verbose: verbose option for debugging
    :return: updated vader disk codes
    """
    # For debugging
    if verbose:
        print(time.value_in(units.yr), stars.mass.value_in(units.MSun))

    closest_approach = periastron_distance(stars)
    # Update collisional radius so that we don't detect this encounter in the next time step
    stars.collisional_radius = 0.49 * closest_approach

    new_codes = []
    truncated = False

    # Check each star
    for i in range(2):
        if disk_codes[i] is None:  # Bright star, no disk code
            new_codes.append(None)
        else:
            truncation_radius = (closest_approach.value_in(units.AU) / 3) *\
                                 numpy.sqrt(stars[i].stellar_mass.value_in(units.MSun)
                                      / stars[1 - i].stellar_mass.value_in(units.MSun)) | units.AU

            R_disk = stars[i].disk_radius  #get_disk_radius(disk_codes[i])
            print "R_disk = {0}, truncation radius={1}".format(R_disk, truncation_radius)

            if truncation_radius < R_disk:
                truncated = True
                print "truncating encounter"
                stars[i].encounters += 1

                old_mass = get_disk_mass(disk_codes[i], R_disk)

                new_disk = truncate_disk(disk_codes[i], truncation_radius)
                new_codes.append(new_disk)

                new_mass = get_disk_mass(new_disk, truncation_radius)

                stars[i].truncation_mass_loss = old_mass - new_mass
                stars[i].disk_mass = new_mass

            else:
                new_codes.append(disk_codes[i])

            # Truncating the "no photoevaporation" disk, if needed
            if truncation_radius < stars[i].disk_size_np:
                stars[i].disk_size_np = truncation_radius
                stars[i].disk_mass_np *= 1.6

    return truncated, new_codes


def truncate_disk(disk, new_radius, density_limit=1E-11):
    """ Truncate a vader disk to new_radius

    :param disk: disk code to truncate
    :param new_radius: new radius of disk
    :param density_limit: density limit for disk boundary
    :return: vader code with disk at new radius
    """

    disk.grid[disk.grid.r >= new_radius].column_density = density_limit | units.g / units.cm**2
    return disk


def evaporate(disk, mass):
    """ Return new size disk after photoevaporation.
        Goes through the disk outside-in removing mass until the needed amount is reached.

    :param disk: vader disk to truncate
    :param mass: mass lost to photoevaporation in MSun
    :return: vader code with disk at new radius
    """

    radius = get_disk_radius(disk).value_in(units.au)

    if radius < 1.:
        return disk

    init_cell = numpy.where(disk.grid.r.value_in(units.au) == radius)[0][0]

    for d, r, a in zip(disk.grid[init_cell::-1].column_density, disk.grid[init_cell::-1].r, disk.grid[init_cell::-1].area):
        cell_mass_msun = d.value_in(units.MSun / (units.AU ** 2)) * a.value_in(units.AU ** 2) | units.MSun

        if mass >= cell_mass_msun:
            #print "go again"
            mass_left = mass - cell_mass_msun
            return evaporate(truncate_disk(disk, r), mass_left)

        else:
            return disk

@timer
def main(N, Rvir, Qvir, alpha, ncells, t_ini, t_end, save_interval, run_number, save_path, dt=2000 | units.yr):

    try:
        float(t_end)
        t_end = t_end | units.Myr
    except TypeError:
        pass

    t = 0 | t_end.unit

    path = "{0}/{1}/".format(save_path, run_number)
    try:
        os.makedirs(path)
        print "Results path created"
    except OSError, e:
        if e.errno != 17:
            raise
        # time.sleep might help here
        pass

    max_stellar_mass = 100 | units.MSun
    stellar_masses = new_kroupa_mass_distribution(N, max_stellar_mass, random=False)
    converter = nbody_system.nbody_to_si(stellar_masses.sum(), Rvir)
    stars = new_fractal_cluster_model(N=N, fractal_dimension=1.6, convert_nbody=converter)
    stars.scale_to_standard(converter, virial_ratio=Qvir)

    stars.stellar_mass = stellar_masses
    stars.encounters = 0  # Counter for dynamical encounters

    # Bright stars: no disks; emit FUV radiation
    bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

    if len(bright_stars) == 0:  # For small tests sometimes we don't get any stars > 1.9MSun, so we add one
        big_star = randint(2, 100)
        stars[0].stellar_mass = big_star | units.MSun
        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        print("Warning: No star with mass > 1.9 MSun generated by the IMF."
              "\nOne star of {0} MSun added to the simulation.".format(big_star))
    bright_stars.bright = True

    print stars.stellar_mass.value_in(units.MSun)

    # Small stars: with disks; radiation from them not considered
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) < 1.9]
    small_stars.bright = False

    small_stars.disk_radius = 100 * (small_stars.stellar_mass.value_in(units.MSun) ** 0.5) | units.AU
    bright_stars.disk_radius = 0 | units.AU

    bright_stars.disk_mass = 0 | units.MSun
    small_stars.disk_mass = 0.1 * small_stars.stellar_mass

    # Initially all stars have the same collisional radius
    stars.collisional_radius = 0.02 | units.parsec

    disk_codes = []
    global disk_codes_indices
    disk_codes_indices = {}  # Using this to keep track of codes later on, for the encounters
    code_channels =[]

    # Create individual instances of vader codes for each disk
    for s in stars:
        if s in small_stars:
            s.code = True
            s_code = initialize_vader_code(s.disk_radius, s.disk_mass, alpha, n_cells=ncells, linear=False)
            

            disk_codes.append(s_code)
            disk_codes_indices[s.key] = len(disk_codes) - 1            

            # Saving these values to keep track of dispersed disks later on
            s.dispersed_disk_mass = 0.01 * s.disk_mass
            s.dispersion_threshold = 1E-11  # Density threshold for dispersed disks
            s.dispersed = False
            s.dispersal_time = t
            s.photoevap_mass_loss = 0 | units.MSun
            s.truncation_mass_loss = 0 | units.MSun

            # Initial values of disks
            s.initial_disk_size = get_disk_radius(s_code)
            s.initial_disk_mass = get_disk_mass(s_code, s.initial_disk_size)

            # Value to keep track of disk sizes and masses as not influenced by photoevaporation
            s.disk_size_np = s.initial_disk_size
            s.disk_mass_np = s.initial_disk_mass

        else:  # Attach None to bright stars' codes
            s.code = False

    # Start gravity code, add all stars
    gravity = ph4(converter)
    gravity.parameters.timestep_parameter = 0.01
    gravity.parameters.epsilon_squared = (100 | units.AU) ** 2
    gravity.particles.add_particles(stars)

    # Enable stopping condition for dynamical encounters
    dynamical_encounter = gravity.stopping_conditions.collision_detection
    dynamical_encounter.enable()

    # Start stellar evolution code, add only massive stars
    stellar = SeBa()
    stellar.parameters.metallicity = 0.02
    stellar.particles.add_particles(bright_stars)
    # Enable stopping on supernova explosion
    detect_supernova = stellar.stopping_conditions.supernova_detection
    detect_supernova.enable()

    # Communication channels
    channel_from_stellar_to_framework = stellar.particles.new_channel_to(stars)
    channel_from_stellar_to_gravity = stellar.particles.new_channel_to(gravity.particles)
    channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars)
    channel_from_framework_to_gravity = stars.new_channel_to(gravity.particles,
                                                             attributes=['collisional_radius'],
                                                             target_names=['radius'])
    channel_from_framework_to_stellar = stars.new_channel_to(stellar.particles)

    channel_from_framework_to_gravity.copy()

    ######## FRIED grid ########
    # Read FRIED grid
    grid = numpy.loadtxt('../friedgrid.dat', skiprows=2)

    # Getting only the useful parameters from the grid (not including Mdot)
    FRIED_grid = grid[:, [0, 1, 2, 4]]
    grid_log10Mdot = grid[:, 5]

    grid_stellar_mass = FRIED_grid[:, 0]
    grid_FUV = FRIED_grid[:, 1]
    grid_disk_mass = FRIED_grid[:, 2]
    grid_disk_radius = FRIED_grid[:, 3]

    E_ini = gravity.kinetic_energy + gravity.potential_energy

    # For keeping track of energy
    E_handle = file('{0}/{1}/energy.txt'.format(save_path, run_number), 'a')
    Q_handle = file('{0}/{1}/virial.txt'.format(save_path, run_number), 'a')
    E_list = []
    Q_list = []

    write_set_to_file(stars,
                      '{0}/{1}/N{2}_t{3}.hdf5'.format(save_path,
                                                      run_number,
                                                      N,
                                                      # Rvir.value_in(units.parsec),
                                                      t.value_in(units.Myr)),
                      'hdf5')

    """try:
        while True:
            stellar.evolve_model()
            print stellar.particles.stellar_type
            if detect_supernova.is_set():
                print "SUPERNOVA EXPLOSION!!!"
    except Exception as exc:
        stellar.stop()"""

    #stellar.evolve_model(100 | units.Myr)
    #print stellar.particles[0].stellar_type == 10 | units.stellar_type

    channel_from_stellar_to_framework.copy()
    channel_from_stellar_to_gravity.copy()
    channel_from_framework_to_stellar.copy()

    # Evolve!
    while t < t_end:
        #print t
        dt = min(dt, t_end - t)

        stellar.evolve_model(t + dt/2)
        channel_from_stellar_to_gravity.copy()
        channel_from_stellar_to_framework.copy()

        E_kin = gravity.kinetic_energy
        E_pot = gravity.potential_energy

        E_list.append([(E_kin + E_pot) / E_ini - 1])
        Q_list.append([-1.0 * E_kin / E_pot])

        gravity.evolve_model(t + dt)

        if dynamical_encounter.is_set():
            #print("encounter")
            #print "before enc stars.disk_radius:"
            #print stars.disk_radius
            channel_from_gravity_to_framework.copy()
            encountering_stars = Particles(particles=[dynamical_encounter.particles(0)[0],
                                                      dynamical_encounter.particles(1)[0]])

            #print stars.key

            # This is to manage encounters involving bright stars (which have no associated vader code)
            try:
                code_index = [disk_codes_indices[encountering_stars[0].key],
                              disk_codes_indices[encountering_stars[1].key]]
                star_codes = [disk_codes[code_index[0]], disk_codes[code_index[1]]]
                print "small - small"
                print "key1: {0}, key2: {1}".format(encountering_stars[0].key, encountering_stars[1].key)
            except KeyError:
                if encountering_stars[0] in bright_stars:
                    print "bright - small"
                    code_index = [None, disk_codes_indices[encountering_stars[1].key]]
                    star_codes = [None, disk_codes[code_index[1]]]
                    print "key1: {0}, key2: {1}".format(encountering_stars[0].key, encountering_stars[1].key)
                elif encountering_stars[1] in bright_stars:
                    print "small - bright"
                    code_index = [disk_codes_indices[encountering_stars[0].key], None]
                    star_codes = [disk_codes[code_index[0]], None]
                    print "key1: {0}, key2: {1}".format(encountering_stars[0].key, encountering_stars[1].key)
                else:
                    print "bright - bright"
                    star_codes = [None, None]
                    print "key1: {0}, key2: {1}".format(encountering_stars[0].key, encountering_stars[1].key)

            #if encountering_stars.get_intersecting_subset_in(stars)[0] in small_stars and encountering_stars.get_intersecting_subset_in(stars)[1] in small_stars:
                #print "before resolve_encounter:"
                #print get_disk_radius(disk_codes[disk_codes_indices[encountering_stars.get_intersecting_subset_in(stars)[0].key]]), \
                #    get_disk_radius(disk_codes[disk_codes_indices[encountering_stars.get_intersecting_subset_in(stars)[1].key]])
                #print get_disk_radius(star_codes[0]), get_disk_radius(star_codes[1])

            truncated, new_codes = resolve_encounter(encountering_stars.get_intersecting_subset_in(stars),
                                                     star_codes,
                                                     gravity.model_time + t_ini)

            if truncated:
                if new_codes[0] is not None and new_codes[1] is not None:
                    #print "small-small"
                    #print "after trunc: {0}, {1}".format(get_disk_radius(disk_codes[code_index[0]]),
                    #                                   get_disk_radius(disk_codes[code_index[1]]))
                    disk_codes[code_index[0]] = new_codes[0]
                    disk_codes[code_index[1]] = new_codes[1]
                    encountering_stars.get_intersecting_subset_in(stars)[0].disk_radius = get_disk_radius(disk_codes[code_index[0]])
                    encountering_stars.get_intersecting_subset_in(stars)[1].disk_radius = get_disk_radius(disk_codes[code_index[1]])
                    #print disk_codes[code_index[0]], new_codes[0]
                    #print "after after trunc: {0}, {1}".format(get_disk_radius(disk_codes[code_index[0]]),
                    #                                   get_disk_radius(disk_codes[code_index[1]]))
                elif new_codes[0] is None and new_codes[1] is not None:
                    #print "big-small"
                    #print "pre trunc: {0}".format(get_disk_radius(disk_codes[code_index[1]]))
                    disk_codes[code_index[1]] = new_codes[1]
                    encountering_stars.get_intersecting_subset_in(stars)[1].disk_radius = get_disk_radius(disk_codes[code_index[1]])
                    #print "post trunc: {0}".format(get_disk_radius(disk_codes[code_index[1]]))
                elif new_codes[0] is not None and new_codes[1] is None:
                    #print "small-big"
                    #print "pre trunc: {0}".format(get_disk_radius(disk_codes[code_index[0]]))
                    disk_codes[code_index[0]] = new_codes[0]
                    encountering_stars.get_intersecting_subset_in(stars)[0].disk_radius = get_disk_radius(disk_codes[code_index[0]])
                    #print "post trunc: {0}".format(get_disk_radius(disk_codes[code_index[0]]))

            #print "after enc stars.disk_radius:"
            #print stars.disk_radius

        # Copy stars' new collisional radii (updated in resolve_encounter) to gravity
        channel_from_framework_to_gravity.copy()
        #print "after evolution collisional_radius:"
        #print gravity.particles.radius.value_in(units.au)

        # Evolve stellar evolution for remaining half time step
        stellar.evolve_model(dt/2)
        channel_from_stellar_to_gravity.copy()
        channel_from_stellar_to_framework.copy()

        # Detect supernova explosion after evolving stellar evolution
        # Delete star that went through supernova explosion
        # Delete all disks within 0.3 pc of the supernova explosion

        # TODO check why supernova stopping condition is not working
        #if detect_supernova.is_set():
        if stellar.particles[0].stellar_type == 10 | units.stellar_type:
            print "SUPERNOVA EXPLOSION!!!"
            channel_from_stellar_to_framework.copy()
            channel_from_gravity_to_framework.copy()
            particles_in_supernova = Particles(particles=detect_supernova.particles(0))
            supernova_star = particles_in_supernova.get_intersecting_subset_in(stars)

            # Parameters from Portegies Zwart 2018
            a = 66.018192
            b = 0.62602783
            c = -0.68226438
            induced_inclination = 0.0  # For now

            for n in small_stars:
                explosion_distance = distance(supernova_star, n)
                r_disk = a * explosion_distance.value_in(units.parsec) ** b * abs(numpy.cos(induced_inclination)) ** c
                if 0. < r_disk < n.disk_radius:
                    new_code = truncate_disk(disk_codes[disk_codes_indices[n.key]], r_disk)
                    disk_codes[disk_codes_indices[n.key]] = new_code
                    n.disk_radius = get_disk_radius(new_code)
                    n.disk_mass = get_disk_mass(new_code, n.disk_radius)
                elif r_disk == 0.:
                    n.disk_radius = 0. | units.au
                    n.disk_mass = 0. | units.MSun
                    n.dispersed = True
                    n.nearby_supernovae = True
                    disk_codes[disk_codes_indices[n.key]].stop()
                    del disk_codes[disk_codes_indices[n.key]]
                    del disk_codes_indices[n.key]


            channel_from_framework_to_gravity.copy()
            del stars[stars.key == supernova_star.key]
            del bright_stars[bright_stars.key == supernova_star.key]

        # Viscous evolution
        evolve_parallel_disks(disk_codes, t + dt)

        # Check disks
        for s, c in zip(small_stars, disk_codes):
            # Check for diverged disks
            if s.code:
                if diverged_disks[c]:  # Disk diverged
                    print "codes len: {0}".format(len(disk_codes))
                    s.dispersed = True
                    s.code = False
                    s.dispersal_time = t
                    c.stop()
                    del disk_codes[disk_codes_indices[s.key]]  # Delete diverged code
                    del disk_codes_indices[s.key]
                    print "deleted diverged code {0}".format(ss.key)
                    print "codes len: {0}".format(len(disk_codes))
                    continue

                # Add accreted mass from disk to host star
                s.stellar_mass += c.inner_boundary_mass_out.value_in(units.MSun) | units.MSun

                # Check for dispersed disks
                disk_density = get_disk_mass(c, s.disk_radius).value_in(units.g) / (numpy.pi * s.disk_radius.value_in(units.cm)**2)
                if get_disk_mass(c, s.disk_radius) <= s.dispersed_disk_mass or s.disk_radius.value_in(units.au) < 0.5 or disk_density <= s.dispersion_threshold:  # Disk has been dispersed
                    #print small_stars
                    s.dispersed = True
                    s.code = False
                    s.dispersal_time = t
                    print "prev: len(disk_codes)={0}, len(disk_code_indices)={1}".format(len(disk_codes), len(disk_codes_indices))
                    disk_codes[disk_codes_indices[s.key]].stop()
                    del disk_codes[disk_codes_indices[s.key]]  # Delete dispersed disk from code list
                    del disk_codes_indices[s.key]
                    print "Star's {0} disk dispersed, deleted code".format(s.key)
                    print "post: len(disk_codes)={0}, len(disk_code_indices)={1}".format(len(disk_codes),
                                                                                         len(disk_codes_indices))
                    continue

            # Update stars disk radius and mass
            s.disk_radius = get_disk_radius(c)
            s.disk_mass = get_disk_mass(c, s.disk_radius)

        # Photoevaporation
        for s in bright_stars:  # For each massive/bright star
            # Calculate FUV luminosity of the bright star, in LSun
            lum = luminosity_fit(s.stellar_mass.value_in(units.MSun))

            for ss in small_stars:
                if ss.dispersed:  # We ignore dispersed disks
                    continue

                #print "continuing. ss.key = {0}".format(ss.key)
                dist = distance(s, ss)
                radiation_ss = radiation_at_distance(lum.value_in(units.erg / units.s),
                                                     dist.value_in(units.cm))

                radiation_ss_G0 = radiation_ss.value_in(units.erg/(units.s * units.cm**2)) / 1.6E-3
                #print(ss.mass.value_in(units.MSun),
                #      radiation_ss_G0,
                #      ss.disk_mass.value_in(units.MJupiter),
                #      ss.disk_radius.value_in(units.AU)
                #      )

                # For the small star, I want to interpolate the photoevaporation mass loss
                # xi will be the point used for the interpolation. Adding star values...
                xi = numpy.ndarray(shape=(1, 4), dtype=float)
                xi[0][0] = ss.stellar_mass.value_in(units.MSun)
                xi[0][1] = radiation_ss_G0
                print "trying {0}".format(ss.key)
                xi[0][3] = get_disk_radius(disk_codes[disk_codes_indices[ss.key]]).value_in(units.AU)
                xi[0][2] = get_disk_mass(disk_codes[disk_codes_indices[ss.key]], xi[0][3] | units.AU).value_in(units.MJupiter)

                # Building the subgrid (of FRIED grid) over which I will perform the interpolation
                subgrid = numpy.ndarray(shape=(8, 4), dtype=float)

                # Finding indices between which ss.mass is located in the grid
                stellar_mass_i, stellar_mass_j = find_indices(grid_stellar_mass, ss.stellar_mass.value_in(units.MSun))
                subgrid[0] = FRIED_grid[stellar_mass_i]
                subgrid[1] = FRIED_grid[stellar_mass_j]

                # Finding indices between which the radiation over the small star is located in the grid
                FUV_i, FUV_j = find_indices(grid_FUV, radiation_ss_G0)
                subgrid[2] = FRIED_grid[FUV_i]
                subgrid[3] = FRIED_grid[FUV_j]

                # Finding indices between which ss.disk_mass is located in the grid
                disk_mass_i, disk_mass_j = find_indices(grid_disk_mass, ss.disk_mass.value_in(units.MJupiter))
                subgrid[4] = FRIED_grid[disk_mass_i]
                subgrid[5] = FRIED_grid[disk_mass_j]

                # Finding indices between which ss.disk_radius is located in the grid
                disk_radius_i, disk_radius_j = find_indices(grid_disk_radius, ss.disk_radius.value_in(units.AU))
                subgrid[6] = FRIED_grid[disk_radius_i]
                subgrid[7] = FRIED_grid[disk_radius_j]

                # Adding known values of Mdot, in the indices found above, to perform interpolation
                Mdot_values = numpy.ndarray(shape=(8, ), dtype=float)
                indices_list = [stellar_mass_i, stellar_mass_j,
                                FUV_i, FUV_j,
                                disk_mass_i, disk_mass_j,
                                disk_radius_i, disk_radius_j]
                for x in indices_list:
                    Mdot_values[indices_list.index(x)] = grid_log10Mdot[x]

                # Interpolate!
                # Photoevaporative mass loss in log10(MSun/yr)
                photoevap_Mdot = interpolate.griddata(subgrid, Mdot_values, xi, method="nearest")

                # Calculate total mass lost due to photoevaporation during dt, in MSun
                total_photoevap_mass_loss = float(numpy.power(10, photoevap_Mdot) * dt.value_in(units.yr)) | units.MSun

                ss.photoevap_mass_loss += total_photoevap_mass_loss

                #print "mass loss: {0}".format(total_photoevap_mass_loss)

                #print "pre evaporate: {0}".format(get_disk_radius(disk_codes[disk_codes_indices[ss.key]]))
                disk_codes[disk_codes_indices[ss.key]] = evaporate(disk_codes[disk_codes_indices[ss.key]],
                                                                   total_photoevap_mass_loss)
                ss.disk_radius = get_disk_radius(disk_codes[disk_codes_indices[ss.key]])
                ss.disk_mass = get_disk_mass(disk_codes[disk_codes_indices[ss.key]], ss.disk_radius)
                #print "AFTER PHOTOEVAP: {0}".format(ss.disk_radius)
                #print "post evaporate: {0}".format(get_disk_radius(disk_codes[disk_codes_indices[ss.key]]))

        #print stars.disk_radius
        #print stars.disk_mass

        if (numpy.around(t.value_in(units.yr)) % save_interval.value_in(units.yr)) == 0.:
            print "saving! at t = {0} Myr".format(t.value_in(units.Myr))
            write_set_to_file(stars,
                              '{0}/{1}/N{2}_t{3}.hdf5'.format(save_path,
                                                          run_number,
                                                          N,
                                                          #Rvir.value_in(units.parsec),
                                                          t.value_in(units.Myr)),
                              'hdf5')

        numpy.savetxt(E_handle, E_list)
        numpy.savetxt(Q_handle, Q_list)

        E_list = []
        Q_list = []


        t += dt


    print "end radii:"
    for d in disk_codes:
        print get_disk_radius(d)
        d.stop()
    print stellar.particles.luminosity
    print stars
    #print small_stars.disk_radius
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
    result.add_option("-a", dest="alpha", type="float", default=5E-3,
                      help="turbulence parameter [%default]")
    result.add_option("-c", dest="ncells", type="int", default=100,
                      help="Number of cells to be used in vader disk [%default]")

    # Time parameters
    result.add_option("-I", dest="t_ini", type="int", default=0 | units.yr,
                      help="initial time [%default]")
    result.add_option("-t", dest="dt", type="int", default=1000 | units.yr,
                      help="dt for simulation [%default]")
    result.add_option("-e", dest="t_end", type="float", default=2 | units.Myr,
                      help="end time of the simulation [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

