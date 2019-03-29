import numpy
import math
from matplotlib import rc
import matplotlib
from matplotlib import pyplot
from amuse.lab import *
from amuse import io


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
matplotlib.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
matplotlib.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels


def distance(star1, star2, center=False):
    """ Return distance between star1 and star2

    :param star1:
    :param star2:
    :param center: if True, just calculate the distance from star1 to (0, 0, 0)
    :return:
    """
    if center:
        return numpy.sqrt(star1.x**2 + star1.y**2 + star1.z**2)

    return numpy.sqrt((star2.x - star1.x)**2 + (star2.y - star1.y)**2 + (star2.z - star1.z)**2)


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


def size_vs_distance_from_star(paths, t_end, N, labels, colors, density):
    """ Plot average distance to bright star (au?) vs disk diameter (au)

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    times = numpy.arange(0.0, 10.05, 0.05)

    for p in paths:
        filename_final = '{0}/N{1}_t{2}.hdf5'.format(p, N, t_end)
        final_stars = io.read_set_from_file(filename_final, 'hdf5', close_file=True)  # Stars at the end of the simulation

        distances_dict = {}
        sizes_dict = {}

        for fs in final_stars:
            distances_dict[fs.key] = []
            sizes_dict[fs.key] = 2 * fs.disk_radius.value_in(units.au)

        for t in times:
            filename = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(filename, 'hdf5', close_file=True)

            small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
            small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                      (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]

            bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

            for ss in small_stars:
                bs_distances = []
                for bs in bright_stars:
                    d = distance(ss, bs).value_in(units.parsec)
                    bs_distances.append(d)

                med = numpy.min(bs_distances)
                distances_dict[ss.key].append(numpy.median(bs_distances))

        all_distances, disk_sizes = [], []

        for key in distances_dict:
            dd = numpy.min(distances_dict[key])
            if math.isnan(dd):
                print "NAN!!!!!!!!!!!!!!!"
            all_distances.append(dd)
            disk_sizes.append(sizes_dict[key])

        sorted_distances, sorted_disk_sizes = zip(*sorted(zip(all_distances, disk_sizes)))
        sorted_distances = list(sorted_distances)
        sorted_disk_sizes = list(sorted_disk_sizes)

        #print sorted_distances, sorted_disk_sizes

        ax.plot(sorted_distances, sorted_disk_sizes)

    ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N=100, c=100, t=10 Myr')
    ax.set_xlabel('Average distance to bright star [parsec]')
    ax.set_ylabel('Disk size [au]')
    pyplot.show()
    #fig.savefig('plot2.png')


def mass_loss_distribution(path, t, N, colors, density):
    fig = pyplot.figure(figsize=(12, 12))
    ax = pyplot.gca()

    # Open t=10.0 Myr file to create colormap
    f = '{0}/N{1}_t{2}.hdf5'.format(path, N, 10.0)
    stars = io.read_set_from_file(f, 'hdf5', close_file=True)

    # Take only the small stars
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
    small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                              (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
    disked_stars = small_stars[small_stars.dispersed == False]

    bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

    mass_loss_trunc = disked_stars.truncation_mass_loss.value_in(units.MSun)
    mass_loss_photoevap = disked_stars.photoevap_mass_loss.value_in(units.MSun)

    zero = numpy.zeros((len(mass_loss_trunc)))
    # Negative values for colormap: mass loss due to dynamical truncations.
    # Positive values for colormap: mass loss due to photoevaporation.
    mass_loss_map = zero - mass_loss_trunc + mass_loss_photoevap

    print mass_loss_map

    minima = min(mass_loss_map)
    maxima = max(mass_loss_map)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.coolwarm)

    #times = numpy.arange(0.0, 10.05, 0.05)

    #for t in times:
    f = '{0}/N{1}_t{2}.hdf5'.format(path, N, t)
    stars = io.read_set_from_file(f, 'hdf5', close_file=True)

    # Take only the small stars
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
    small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                              (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
    disked_stars = small_stars[small_stars.dispersed == False]

    mass_loss_trunc = disked_stars.truncation_mass_loss.value_in(units.MSun)
    mass_loss_photoevap = disked_stars.photoevap_mass_loss.value_in(units.MSun)

    zero = numpy.zeros((len(mass_loss_trunc)))
    plot_mass_loss_map = zero - mass_loss_trunc + mass_loss_photoevap

    color_map = []

    for v in plot_mass_loss_map:
        print mapper.to_rgba(v)

    ml = ax.scatter(disked_stars.x.value_in(units.parsec), disked_stars.y.value_in(units.parsec),
               s=disked_stars.disk_radius.value_in(units.au),
               c=plot_mass_loss_map, cmap=matplotlib.cm.coolwarm, alpha=0.5)

    #ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N={0}, c=100, t={1} Myr'.format(N, t))
    ax.set_xlabel('x [parsec]')
    ax.set_ylabel('y [parsec]')
    pyplot.colorbar(mappable=ml)
    pyplot.show()
    # fig.savefig('plot2.png')



def distance_from_center(paths, t, N, labels, colors, density):
    """ Plot disk size vs distance from cluster center

    :param paths: list of paths of files to plot
    :param t: time to use for the plot
    :param N: number of stars
    :param labels: labels for plots
    :param colors: colors for plot lines
    :param density: density threshold for disk surface
    """

    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for p, l, c in zip(paths, labels, colors):
        f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        disked_stars = small_stars[small_stars.dispersed == False]

        distances = []
        disk_sizes = []

        for ds in disked_stars:
            d = distance(ds, ds, center=True)
            distances.append(d.value_in(units.parsec))
            disk_sizes.append(ds.disk_radius.value_in(units.au))

        sorted_distances, sorted_disk_sizes = zip(*sorted(zip(distances, disk_sizes)))
        sorted_distances = list(sorted_distances)
        sorted_disk_sizes = list(sorted_disk_sizes)

        ax.plot(sorted_distances, sorted_disk_sizes, label=l, color=c)

    ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N={0}, c=100, t={1} Myr'.format(N, t))
    ax.set_xlabel('Distance to cluster center [parsec]')
    ax.set_ylabel('Disk size [au]')
    pyplot.show()
    # fig.savefig('plot2.png')




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


def plot_cluster(path, t, N, colors, density):
    """ Plot star positions and disk sizes

    :param paths: list of paths of files to plot
    :param t: time to use for the plot
    :param N: number of stars
    :param labels: labels for plots
    :param colors: colors for plot lines
    :param density: density threshold for disk surface
    """

    fig = pyplot.figure(figsize=(12, 12))
    ax = pyplot.gca()

    for c in colors:
        f = '{0}/N{1}_t{2}.hdf5'.format(path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        disked_stars = small_stars[small_stars.dispersed == False]

        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

        ax.scatter(disked_stars.x.value_in(units.parsec), disked_stars.y.value_in(units.parsec),
                   s=disked_stars.disk_radius.value_in(units.au),
                   color=c, alpha=0.5)
        ax.scatter(bright_stars.x.value_in(units.parsec), bright_stars.y.value_in(units.parsec),
                   marker='*', color='k', s=100)



    #ax.legend(loc='upper right', fontsize=20)
    ax.set_title('N={0}, c=100, t={1} Myr'.format(N, t))
    ax.set_xlabel('x [parsec]')
    ax.set_ylabel('y [parsec]')
    pyplot.show()
    # fig.savefig('plot2.png')


def main(run_number, save_path, time, N, distribution, ncells, density):

    files = ['results/king_N100_c100_3/0/N100_t10.0.hdf5',
             'results/plummer_N100_c100_3/0/N100_t10.0.hdf5']#,
             #'results/fractal_N100_c50/0/N100_t0.hdf5']

    paths = ['results/king_N{0}_c{1}_3/0/'.format(N, ncells),
             'results/plummer_N{0}_c{1}_3/0/'.format(N, ncells)]#,
             #'results/fractal_N{0}_c{1}_3/0/']

    labels = ["King", "Plummer", "Fractal"]
    plot_colors = ["#ca5670", "#638ccc", "#c57c3c", "#72a555", "#ab62c0"]

    #size_vs_mass(files, labels, plot_colors, density, N, ncells, 10.0)
    #size_vs_distance_from_star(paths, time, N, labels, plot_colors, density)
    #distance_from_center(paths, time, N, labels, plot_colors, density)

    path = 'results/{0}_N{1}_c{2}_3/0/'.format(distribution, N, ncells)
    #plot_cluster(path, time, N, plot_colors, density)
    mass_loss_distribution(path, time, N, plot_colors, density)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation parameters
    result.add_option("-n", dest="run_number", type="int", default=0,
                      help="run number [%default]")
    result.add_option("-s", dest="save_path", type="string", default='.',
                      help="path to save the results [%default]")
    result.add_option("-t", dest="time", type="float", default='10.0',
                      help="time to use for plots [%default]")

    # Cluster parameters
    result.add_option("-N", dest="N", type="int", default=100,
                      help="number of stars [%default]")
    result.add_option("-p", dest="distribution", type="string", default='plummer',
                      help="spatial distribution to plot [%default]")

    # Disk parameters
    result.add_option("-c", dest="ncells", type="int", default=100,
                      help="Number of cells to be used in vader disk [%default]")
    result.add_option("-d", dest="density", type="float", default=1E-9,
                      help="Density limit for disk surface [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

