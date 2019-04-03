from amuse.lab import *
import numpy
from random import randint
from amuse import io
from amuse.couple.bridge import Bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.ic.kingmodel import new_king_model
from scipy import interpolate
import multiprocessing
from decorators import timer
import os

# All disk functions are here
from disks import *

@timer
def main(N, Rvir, Qvir, dist, alpha, ncells, t_ini, t_end, save_interval, run_number, save_path, dt=2000 | units.yr):
#def main(N, Rvir, Qvir, dist, alpha, ncells, t_ini, t_end, save_interval, run_number, save_path, dust_density, dt=2000 | units.yr): # MW

    from matplotlib import pyplot


    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    star = Particles(mass=1. | units.MSun)
    initial_disk_mass = 0.1 * star.mass.value_in(units.MSun) | units.MSun
    initial_disk_radius = 100 * (star.mass.value_in(units.MSun) ** 0.5) | units.AU
    print "initial radius: ", initial_disk_radius

    disk = initialize_vader_code(initial_disk_radius, initial_disk_mass, 5E-3, linear=False)
    ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), label="initial", c='r', marker='x')


    t_end = 10. | units.Myr
    dt = 0.1 | units.Myr

    t = 0. | units.Myr

    cumulative_mass_loss = 0.0 | units.MJupiter

    while t < t_end:
        print "t = ", t
        r = get_disk_radius(disk)
        print "disk radius before viscous: ", get_disk_radius(disk)
        print "disk mass before viscous: ", get_disk_mass(disk, r)
        ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), label='before viscous', c='b')
        ax.axvline(get_disk_radius(disk).value_in(units.au), c='b')

        evolve_single_disk(disk, t + dt)
        ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm ** 2), label="after viscous", c='green')
        ax.axvline((get_disk_radius(disk).value_in(units.au)), c='green')
        r = get_disk_radius(disk)
        print "disk radius after viscous: ", get_disk_radius(disk)
        print "disk mass after viscous: ", get_disk_mass(disk, r)

        cumulative_mass_loss += 0.05 | units.MJupiter

        if cumulative_mass_loss >= 0.01 * initial_disk_mass:
            print "disk is gone"
            break
        else:
            disk = evaporate(disk, 0.05 | units.MJupiter)
            if disk is None:
                print "disk is gone"
                break

        ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm ** 2), ls='--', label="after photoevap", c='orange')
        ax.axvline((get_disk_radius(disk).value_in(units.au)), c='orange')

        r = get_disk_radius(disk)
        print "disk radius after photoevap: ", get_disk_radius(disk)
        print "disk mass after photoevap: ", get_disk_mass(disk, r)
        #break

        r = get_disk_radius(disk)
        t += dt

    ax.legend(loc='best')
    pyplot.show()



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
    result.add_option("-p", dest="dist", type="string", default="plummer",
                      help="spatial distribution [%default]")

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

