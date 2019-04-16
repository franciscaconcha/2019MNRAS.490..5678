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

import matplotlib
from matplotlib import pyplot
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22, })
rc('text', usetex=True)
rc('axes', labelsize=22)  # fontsize of the x and y labels
matplotlib.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
matplotlib.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels

# All disk functions are here
from disks import *


def clarke2007_mdot(Rd):
    #if clarke07 benchmark                                                                                                                  
    if(Rd >= 100.0):
        thismdot=5.1e-8*(Rd/100.)
    else:
        if(Rd>=60. and Rd < 100):
            thismdot=9.9e-8*(420./Rd)*numpy.exp(-210.0/Rd)
        else:
            if(Rd >= 40. and Rd < 60.):
                thismdot = 1.e-8*numpy.exp((Rd-40.)/27.)
            else:
                thismdot = 1.e-8*numpy.exp((Rd-40.)/9.)

    return thismdot | (units.MSun / units.yr)


def clarke2007_test(t_end, mode):
    # Case A
    mass=0.1 | units.MSun
    Rc=10.
    SigmaC=mass/2.0/numpy.pi/(Rc)**2/(1.0-numpy.exp(-60./Rc))
    #initial disc outer edge
    redgeinitA=60.0 | units.AU
    #initial accretion rate: 10**(-6.3)msol/yr

    diskA = initialize_vader_code(redgeinitA, mass, 5E-3, r_min=0.5 | units.AU, r_max=800 | units.AU, n_cells=100, linear=False)


    # Case B
    mass=0.1 | units.MSun
    Rc=100.
    SigmaC=mass/2.0/numpy.pi/(Rc)**2/(1.0-numpy.exp(-222./Rc))
    #initial disc outer edge
    redgeinitB=222.0 | units.AU

    diskB = initialize_vader_code(redgeinitB, mass, 5E-3, r_min=0.5 | units.AU, r_max=800 | units.AU, n_cells=100, linear=False)

    dt = 0.05 | units.Myr

    t = 0. | units.Myr

    cumulative_mass_loss = 0.0 | units.MJupiter

    caseA = []
    caseB = []
    times = []

    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    j = 0

    while t < t_end:
        print "t = ", t

        evolve_single_disk(diskA, t + dt)
        evolve_single_disk(diskB, t + dt)

        if mode=="mass":
            RdA = get_disk_radius_mass(diskA)
            RdB = get_disk_radius_mass(diskB)
        elif mode=="density":
            RdA = get_disk_radius_density(diskA)
            RdB = get_disk_radius_density(diskB)
        else:
            RdA = get_disk_radius_av(diskA)
            RdB = get_disk_radius_av(diskB)

        mdotA = clarke2007_mdot(RdA.value_in(units.AU))
        mdotB = clarke2007_mdot(RdB.value_in(units.AU))

        masslossA = mdotA * dt
        masslossB = mdotB * dt

        diskA = evaporate(diskA, masslossA, mode=mode)
        diskB = evaporate(diskB, masslossB, mode=mode)

        if mode=="mass":
            RdA = get_disk_radius_mass(diskA)
            RdB = get_disk_radius_mass(diskB)
        elif mode=="density":
            RdA = get_disk_radius_density(diskA)
            RdB = get_disk_radius_density(diskB)
        else:
            RdA = get_disk_radius_av(diskA)
            RdB = get_disk_radius_av(diskB)

        caseA.append(RdA.value_in(units.au))
        caseB.append(RdB.value_in(units.au))
        times.append(t)
        
        #save to file!
        print caseA
        print caseB

        #pyplot.plot(diskA.grid.column_density)

        t += dt

    return caseA, caseB, times


@timer
def main(N, Rvir, Qvir, dist, alpha, ncells, t_ini, t_end, save_interval, run_number, save_path, dt=2000 | units.yr):
#def main(N, Rvir, Qvir, dist, alpha, ncells, t_ini, t_end, save_interval, run_number, save_path, dust_density, dt=2000 | units.yr): # MW

    from matplotlib import pyplot


    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    """times, caseA, caseB = clarke2007_test(2 | units.Myr)

    print caseA
    print caseB

    ax.plot(times, caseA, label="Case A")
    ax.plot(times, caseB, label="Case B")

    pyplot.legend()
    pyplot.savefig('figures/tests/testclarke2007-2.png'.format(t))
    #pyplot.show()"""

    stars = Particles(mass=[1., 30.] | units.MSun)

    star = stars[0]
    brightstar = stars[1]

    initial_disk_mass = 0.1 * star.mass.value_in(units.MSun) | units.MSun
    initial_disk_radius = 100 * (star.mass.value_in(units.MSun) ** 0.5) | units.AU
    print "initial radius: ", initial_disk_radius

    disk = initialize_vader_code(initial_disk_radius, initial_disk_mass, 5E-3, linear=False)
    initial = disk.grid.column_density.value_in(units.g / units.cm**2)
    #ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2),
    #         c='r', lw=3)
    ax.set_xlabel('Radius [au]')
    ax.set_ylabel(r'Column density [g / cm$^2$]')

    t_end = 10 | units.Myr
    dt = 0.05 | units.Myr

    t = 0. | units.Myr

    cumulative_mass_loss = 0.0 | units.MJupiter

    radii_density, radii_mass, radii_av = [], [], []
    j = 0

    while t < t_end:
        print "t = ", t
        #r = get_disk_radius(disk)
        #print "Before viscous: ", get_disk_radius(disk), get_disk_mass(disk, r)
        #ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2),
        #          label='', c='b')
        #ax.axvline(get_disk_radius(disk).value_in(units.au), c='b')

        #disk = truncate_disk(disk, 100 | units.au)
        #ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm ** 2),
        #          c='darkorange', lw=3, ls='--')

        #pyplot.show()
        #return 0

        evolve_single_disk(disk, t + dt)

        #ax.axvline((get_disk_radius(disk).value_in(units.au)), c='green', ls=":")
        print "AFTER VISCOUS: "
        r = get_disk_radius_density(disk)
        print "Density: ", get_disk_radius_density(disk), get_disk_mass(disk, r)
        r = get_disk_radius_mass(disk)
        print "Mass: ", get_disk_radius_mass(disk), get_disk_mass(disk, r)
        r = get_disk_radius_av(disk)
        print "Average: ", get_disk_radius_av(disk), get_disk_mass(disk, r)

        cumulative_mass_loss += 0.05 | units.MJupiter

        if cumulative_mass_loss >= 0.01 * initial_disk_mass:
            print "disk is gone"
            break
        else:
            mdot = clarke2007_mdot(get_disk_radius_mass(disk).value_in(units.au))
            disk = evaporate(disk, mdot * dt)
            if disk is None:
                print "disk is gone"
                break

        ax.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm ** 2),
                  c='k', lw=2)

        #r = get_disk_radius(disk)
        print "AFTER PHOTOEVAP: "
        r = get_disk_radius_density(disk)
        radii_density.append(r.value_in(units.au))
        print "Density: ", get_disk_radius_density(disk), get_disk_mass(disk, r)
        r = get_disk_radius_mass(disk)
        radii_mass.append(r.value_in(units.au))
        print "Mass: ", get_disk_radius_mass(disk), get_disk_mass(disk, r)
        r = get_disk_radius_av(disk)
        radii_av.append(r.value_in(units.au))
        print "Average: ", get_disk_radius_av(disk), get_disk_mass(disk, r)
        #break
        print "*"

        print zip(disk.grid.column_density.value_in(units.g / units.cm**2), disk.grid.r.value_in(units.au))

        ax.axvline((get_disk_radius_mass(disk).value_in(units.au)), color='orange', label='Mass')
        ax.axvline((get_disk_radius_density(disk).value_in(units.au)), color='dodgerblue', label='Density')
        ax.axvline((get_disk_radius_av(disk).value_in(units.au)), color='pink', label='Average')

        ax.plot(disk.grid.r.value_in(units.au), numpy.ones(disk.grid.r.shape), marker='o', color='red')

        ax.legend(loc='best')
        pyplot.savefig('figures/radius_tests3/plot{0}.png'.format(j))
        pyplot.cla()        

        t += dt
        j += 1

    pyplot.plot(radii_mass, color='red', label='Mass', lw=2)
    pyplot.plot(radii_density, color='blue', label='Density', lw=2)
    pyplot.plot(radii_av, color='orange', label='Average', lw=2)
    ax.legend(loc='best')
    ax.set_xlabel('time')
    ax.set_ylabel('radius')
    pyplot.savefig('figures/radius_tests3/radii.png')
    #pyplot.show()



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

