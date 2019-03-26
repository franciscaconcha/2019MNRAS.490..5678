from matplotlib import rc
import matplotlib as mpl
from matplotlib import pyplot
from amuse.lab import *
import numpy
from amuse import io

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
mpl.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
mpl.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels


def size_vs_mass(files, labels, colors, density, N, ncells, t):
    """ Plot disk diameter (au) vs mass (MJup)

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for f, l, c in zip(files, labels, colors):
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) / (numpy.pi * small_stars.disk_radius.value_in(units.au)**2) > density]

        sizes, masses = 2 * small_stars.disk_radius.value_in(units.au), small_stars.disk_mass.value_in(units.MJupiter)

        ax.scatter(sizes, masses, s=100 * small_stars.stellar_mass.value_in(units.MSun),
                   c=c, alpha=0.5, label=l)

    ax.legend(loc='upper left', fontsize=20)
    ax.set_title(r'N_*={0}, '.format(N) + r'n_{cells}=' + '{0}, t={1} Myr'.format(ncells, t))
    ax.set_xlabel('Disk size [au]')
    ax.set_ylabel(r'Disk mass [$M_{Jup}$]')
    pyplot.show()
    #fig.savefig('plot2.png')


def distance_from_star(files, labels, colors):
    """ Plot average distance to bright star (au?) vs disk diameter (au)

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for f, l, c in zip(files, labels, colors):
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]

        sizes, masses = 2 * small_stars.disk_radius.value_in(units.au), small_stars.disk_mass.value_in(units.MJupiter)

        ax.scatter(sizes, masses, s=100 * small_stars.stellar_mass.value_in(units.MSun),
                   c=c, alpha=0.5, label=l)

    ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N=100, c=50, different times')
    ax.set_xlabel('Disk size [au]')
    ax.set_ylabel(r'Disk mass [$M_{Jup}$]')
    pyplot.show()
    #fig.savefig('plot2.png')


def surviving_disks(paths, N, cells, labels, colors):
    """ Plot percentage of surviving disks at different times.

    :param path: path to results file
    :param N: number of stars, for results paths
    :param c: number of disk cells, for results paths
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] #range(3)  # Time in Myr
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for p, l, c in zip(paths, labels, colors):
        total_disks = 0
        disk_p = []
        for t in times:
            f = '{0}N{1}_c{2}/0/N{1}_t{3}.hdf5'.format(p, N, cells, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            # Take only the small stars
            small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0:
                total_disks = len(disked_stars)

            disk_p.append(len(disked_stars)/total_disks)

        ax.plot(times, disk_p)

    ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N=100, c=50, different times')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'Fraction of disks remaining')
    ax.set_xlim(left=0.0)
    ax.set_ylim(0, 1)
    pyplot.show()
    #fig.savefig('plot2.png')




def cdfs(files, labels, colors):
    """ Plot cumulative distributions of disk sizes (au).

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for f, l, c in zip(files, labels, colors):
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]

        sizes, masses = 2 * small_stars.disk_radius.value_in(units.au), small_stars.disk_mass.value_in(units.MJupiter)

        ax.scatter(sizes, masses, s=100 * small_stars.stellar_mass.value_in(units.MSun),
                   c=c, alpha=0.5, label=l)

    ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N=100, c=50, different times')
    ax.set_xlabel('Disk size [au]')
    ax.set_ylabel(r'Disk mass [$M_{Jup}$]')
    pyplot.show()
    #fig.savefig('plot2.png')



def main(run_number, save_path, N, ncells, density):
    files = ['results/king_N100_c100_3/0/N100_t10.0.hdf5',
             'results/plummer_N100_c100_3/0/N100_t10.0.hdf5']#,
             #'results/fractal_N100_c50/0/N100_t0.hdf5']

    labels = ["King", "Plummer", "Fractal"]
    plot_colors = ["#638ccc", "#ca5670", "#c57c3c", "#72a555", "#ab62c0"]

    size_vs_mass(files, labels, plot_colors, density, N, ncells, 10.0)
    #distance_from_star()

    #paths = ['results/king_',
    #         'results/king_',
    #         'results/king_']

    #surviving_disks(paths, 100, 50, labels, plot_colors)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation parameters
    result.add_option("-n", dest="run_number", type="int", default=0,
                      help="run number [%default]")
    result.add_option("-s", dest="save_path", type="string", default='.',
                      help="path to save the results [%default]")

    # Cluster parameters
    result.add_option("-N", dest="N", type="int", default=100,
                      help="number of stars [%default]")

    # Disk parameters
    result.add_option("-c", dest="ncells", type="int", default=100,
                      help="Number of cells to be used in vader disk [%default]")
    result.add_option("-d", dest="density", type="float", default=5E-7,
                      help="Density limit for disk surface [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

