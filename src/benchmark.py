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
    # if clarke07 benchmark
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


def clarke2007_test(t_end, radius, Rc):
    mass = 0.1 | units.MSun
    SigmaC = mass/2.0/numpy.pi/(Rc)**2/(1.0-numpy.exp(-radius/Rc))

    #initial accretion rate: 10**(-6.3)msol/yr

    disk = initialize_vader_code(radius | units.AU, mass, 5E-3, r_min=0.5 | units.AU, r_max=800 | units.AU, n_cells=100, linear=False)

    dt = 0.05 | units.Myr

    t = 0. | units.Myr

    radii, times = []

    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    j = 0

    while t < t_end:
        print "t = ", t

        evolve_single_disk(disk, t + dt)

        Rd = get_disk_radius(disk)
        mdot = clarke2007_mdot(Rd.value_in(units.AU))
        massloss = mdot * dt

        disk = evaporate(disk, massloss)

        Rd = get_disk_radius(disk)

        radii.append(Rd.value_in(units.au))
        times.append(t)

        print radii

        t += dt

    return radii, times


def main(radius, Rc, filename):

    from matplotlib import pyplot

    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    radii, times = clarke2007_test(2 | units.Myr, radius, Rc)

    ax.plot(times, radii, label=radius)

    pyplot.legend()
    pyplot.savefig('figures/tests/{0}.png'.format(filename))
    #pyplot.show()


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation parameters
    result.add_option("-r", dest="radius", type="float", default=60.,
                      help="radius [%default]")
    result.add_option("-c", dest="Rc", type="float", default=10.,
                      help="Rc [%default]")
    result.add_option("-f", dest="filename", type="string", default='benchmark',
                      help="filename to save figure [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

