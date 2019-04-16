from amuse.lab import *
import numpy
import multiprocessing

# Workaround for now
global diverged_disks, disk_codes_indices
diverged_disks = {}
disk_codes_indices = {}


def column_density(grid, r0, mass, lower_density=1E-12 | units.g / units.cm**2):
    r = grid.value_in(units.AU) | units.AU
    rd = r0
    Md = mass

    print r, r0

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
    #import time
    #print "Starting processes... n_cpu = {0}".format(n_cpu)
    #startt = time.time()
    #print("Start loop")
    #end = time.time()
    for i in range(len(codes)):
        #p = multiprocessing.Process(name=str(i), target=evolve_single_disk, args=(codes[i], dt, ))
        #processes.append(p)
        #p.start()

	#start = time.time()
	#print("code", i)
	#end = time.time()
	evolve_single_disk(codes[i],dt)
        #print(end - start)
    #endt = time.time()
    #print("End loop",len(codes))
    #print(endt - startt)
        #th = threading.Thread(target=evolve_single_disk, args=[codes[i], dt])
        #th.daemon = True
        #threads.append(th)
        #th.start()

    #for t in threads:
    #    t.join()

    #for p in processes:
    #    p.join()

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


def get_disk_radius_mass(disk, f=0.95):

    Mtot = (disk.grid.area * disk.grid.column_density).sum()
    Mcum = 0. | units.MSun

    edge = -1

    for i in range(len(disk.grid.r)):
        Mcum += disk.grid.area[i] * disk.grid.column_density[i]

        if Mcum > Mtot * f:
            edge = i
            break

    return disk.grid.r[edge].value_in(units.au) | units.au


def get_disk_radius_density(disk, density_limit=1E-10):
    """ Calculate the radius of a disk in a vader grid.
    :param disk: Disk to calculate radius on.
    :param density_limit: Density limit to designate disk border.
    :return: Disk radius in units.AU
    """
    prev_r = disk.grid[0].r

    for i in range(len(disk.grid.r)):
        cell_density = disk.grid[i].column_density.value_in(units.g / units.cm ** 2)
        if cell_density < density_limit:
            return prev_r.value_in(units.au) | units.au
        prev_r = disk.grid[i].r

    return prev_r.value_in(units.au) | units.au


def get_disk_radius_av(disk, density_limit=1E-12):
    """ Calculate disk radius using average of cells around density jump. 
    """
    prev_r = 0
    center = 0
    diff_threshold = 1E4

    for i in range(1, len(disk.grid.r)):
        diff = disk.grid[prev_r].column_density.value_in(units.g / units.cm**2) - disk.grid[i].column_density.value_in(units.g / units.cm**2)
        if diff >= diff_threshold:
            center = i
            break
        prev_r = i

    sum = 0.
    cells = 10

    for i in range(center - cells, center + cells):
        sum += disk.grid.r[i].value_in(units.au)

    return sum / len(range(center - 5, center + 5)) | units.au


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


def truncate_disk(disk, new_radius, density_limit=1E-11):
    """ Truncate a vader disk to new_radius

    :param disk: disk code to truncate
    :param new_radius: new radius of disk
    :param density_limit: density limit for disk boundary
    :return: vader code with disk at new radius
    """

    disk.grid[disk.grid.r > new_radius].column_density = density_limit | units.g / units.cm**2
    return disk


def evaporate(disk, mass_to_remove, mode="mass"):
    """ Return new size disk after photoevaporation.
        Goes through the disk outside-in removing mass until the needed amount is reached.

    :param disk: vader disk to truncate
    :param mass_to_remove: mass lost to photoevaporation in MSun
    :return: vader code with disk at new radius
    """

    if mode=="mass":
        radius = get_disk_radius_mass(disk).value_in(units.au)
    elif mode=="density":
        radius = get_disk_radius_density(disk).value_in(units.au)
    else:
        radius = get_disk_radius_av(disk).value_in(units.au)
    #print radius

    init_cell = numpy.where(disk.grid.r.value_in(units.au) == radius)[0][0]

    swiped_mass = 0.0 | mass_to_remove.unit

    for i in range(init_cell)[::-1]:
        r = disk.grid[i].r
        d = disk.grid[i].column_density
        a = disk.grid[i].area

        cell_mass_msun = d.value_in(mass_to_remove.unit / (units.AU ** 2)) * a.value_in(units.AU ** 2) | mass_to_remove.unit
        swiped_mass += cell_mass_msun
        #print "swiped mass: {0}, cell_mass: {1}, to_remove: {2} [Mjup]".format(swiped_mass, cell_mass_msun, mass_to_remove)

        if swiped_mass < mass_to_remove:
        #    print "continuing"
            continue
        else:
            if i == 0:
                return None
            else:
                #print disk.grid.r.value_in(units.au)
                #print "truncating disk at ", r.value_in(units.au)
                return truncate_disk(disk, r)
