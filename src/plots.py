import numpy
import math
from matplotlib import rc
import matplotlib
from matplotlib import pyplot
from amuse.lab import *
from amuse import io
import matplotlib.lines as mlines


#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22, })
#rc('text', usetex=True)
#rc('axes', labelsize=22)  # fontsize of the x and y labels
#matplotlib.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
#matplotlib.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels


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


def luminosity_fit(mass):  # For G0 calculation
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


def radiation_at_distance(rad, R):
#def radiation_at_distance(rad, R, dust_density, FUV_absorb_coeff):
    """ Return radiation rad at distance R

    :param rad: total radiation of star in erg/s
    :param R: distance in cm
    :return: radiation of star at distance R, in erg * s^-1 * cm^-2
    """
    return rad / (4 * numpy.pi * R**2) | (units.erg / (units.s * units.cm**2))
    #return rad / (4 * numpy.pi * R**2) * numpy.exp(R*dust_density*FUV_absorb_coeff) | (units.erg / (units.s * units.cm**2)) # MW


def calculate_g0(stars):  # Only necessary for now because I was not saving it in the results, but now I am!
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
    bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

    rad = {}

    for s in small_stars:
        rad[s.key] = 0.0

    for s in bright_stars:  # For each massive/bright star
        # Calculate FUV luminosity of the bright star, in LSun
        lum = luminosity_fit(s.stellar_mass.value_in(units.MSun))

        for ss in small_stars:
            # TODO this check might not be necessary here
            if ss.dispersed:  # We ignore dispersed disks
                continue

            # print "continuing. ss.key = {0}".format(ss.key)
            dist = distance(s, ss)
            radiation_ss = radiation_at_distance(lum.value_in(units.erg / units.s),
                                                 dist.value_in(units.cm)
                                                 # , dust_density.value_in(units.g/units.cm**3.), 56186.4102564,# units.cm**2./units.g MW
                                                 )

            rad[ss.key] += radiation_ss.value_in(units.erg / (units.s * units.cm ** 2)) / 1.6E-3

    return rad


def g0_in_time(open_path, save_path, N, i):
    fig = pyplot.figure(figsize=(12, 6))#, gridspec_kw={"width_ratios": [12, 1]})
    ax = pyplot.gca()
    ax.set_title(r'$G_0$ in time')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$G_0$ [erg/cm^2]')

    times = numpy.arange(0.0, 10.05, 0.05)

    g0s = []

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        g0 = calculate_g0(stars)
        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        s = stars[i]

        g0s.append(g0[s.key])

    ax.plot(times, g0s)


def size_vs_mass(files, labels, colors, density, N, ncells, t):
    """ Plot disk diameter (au) vs mass (MJup)

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    for f, l, c in zip(files, labels, colors):
        print "svm opening ", f
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        disked_stars2 = small_stars[small_stars.dispersed == False]
        disked_stars = disked_stars2[disked_stars2.photoevap_mass_loss < disked_stars2.initial_disk_mass]

        print len(disked_stars)

        sizes, masses = 2 * disked_stars.disk_radius.value_in(units.au), disked_stars.disk_mass.value_in(units.MSun)

        ax.scatter(sizes, masses, s=400 * disked_stars.stellar_mass.value_in(units.MSun),
                   c=c, alpha=0.8, label=l)

        n = disked_stars.stellar_mass.value_in(units.MSun)
        positions = [(1.1 * sizes[0], -masses[0]),
                     (sizes[len(n)/2 - 1]*1.05, masses[len(n)/2 - 1]),
                     (sizes[len(n)-1]*1.15, masses[len(n)-1])]

        j = 0

        for i, txt in enumerate(n):
            if i == 0 or i == len(n)-1 or i == len(n)/2 - 1:
                ax.annotate(r'{0:.2} $M_\odot$'.format(txt), (sizes[i], masses[i]),
                            xytext=positions[j])
                j += 1

        bs1 = mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                            markersize=16 * n[0], label=r'{0:.2} $M_\odot$'.format(n[0]),
                            alpha=0.8)
        bs2 = mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                            markersize=16 * n[len(n)/2 - 1], label=r'{0:.2} $M_\odot$'.format(n[len(n)/2 - 1]),
                            alpha=0.9)
        bs3 = mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                            markersize=16 * n[len(n) - 1], label=r'{0:.2} $M_\odot$'.format(n[len(n) - 1]),
                            alpha=0.8)
        #ax.legend(handles=[bs1, bs2, bs3], loc='best', fontsize=14)

    #ax.legend(loc='upper left', fontsize=20)
    ax.set_title(r'$N_*=${0}, '.format(N) + 't={0} Myr'.format(t))
    ax.set_xlabel('Disk size [au]')
    ax.set_ylabel(r'Disk mass [$M_{\odot}$]')
    pyplot.show()
    #fig.savefig('plot2.png')


def size_vs_distance_from_star(paths, t, N, labels, colors, density):
    """ Plot average distance to bright star (au?) vs disk diameter (au)

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.gca()

    times = numpy.arange(0.0, 10.05, 0.05)

    for p in paths:
        filename = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
        stars = io.read_set_from_file(filename, 'hdf5', close_file=True)

        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        disked_stars2 = small_stars[small_stars.dispersed == False]
        disked_stars = disked_stars2[disked_stars2.photoevap_mass_loss < disked_stars2.initial_disk_mass]

        disk_sizes = disked_stars.disk_radius.value_in(units.au)
        mass_loss_photoevap = disked_stars.photoevap_mass_loss.value_in(units.MJupiter)
        mass_loss_trunc = disked_stars.truncation_mass_loss.value_in(units.MJupiter)

        most_massive_star = bright_stars[0]

        for bs in bright_stars:
            if bs.stellar_mass > most_massive_star.stellar_mass:
                most_massive_star = bs

        distances = []

        for ds in disked_stars:
            distances.append(distance(most_massive_star, ds).value_in(units.parsec))

        ax.scatter(distances, disk_sizes, s=mass_loss_photoevap, alpha=0.5)

    #ax.legend(loc='upper left', fontsize=20)
    ax.set_title(r'Distance to most ionizing star, t={0} Myr'.format(t))
    ax.set_xlabel('Distance to star [parsec]')
    ax.set_ylabel(r'Disk size [au]')

    pyplot.show()



def mass_loss_distribution(open_path, save_path, t, N, colors, density):
    from matplotlib.colors import LinearSegmentedColormap
    fig = pyplot.figure(figsize=(13, 12))
    ax = pyplot.gca()
    #fig, ax = pyplot.subplots(1, 2, figsize=(13, 12))#, gridspec_kw={"width_ratios": [12, 1]})
    ax.set_title('N={0}, c=100, t={1} Myr'.format(N, t))
    ax.set_xlabel('x [parsec]')
    ax.set_ylabel('y [parsec]')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('square')
    #pyplot.axes(ax).set_aspect('equal', adjustable='box')

    # Open t=10.0 Myr file to create colormap
    f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, 10.0)
    stars = io.read_set_from_file(f, 'hdf5', close_file=True)

    # Take only the small stars
    small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
    small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                              (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
    disked_stars = small_stars[small_stars.dispersed == False]

    final_mass_loss_trunc = disked_stars.truncation_mass_loss.value_in(units.MSun)
    final_mass_loss_photoevap = disked_stars.photoevap_mass_loss.value_in(units.MSun)

    #zero = numpy.zeros((len(mass_loss_trunc)))
    # Negative values for colormap: mass loss due to dynamical truncations.
    # Positive values for colormap: mass loss due to photoevaporation.
    #mass_loss_map = zero - mass_loss_trunc + mass_loss_photoevap

    #minima = max(mass_loss_trunc)
    #maxima = max(mass_loss_photoevap)
    #print minima, maxima, min(mass_loss_photoevap)

    #colormap = matplotlib.cm.coolwarm

    #norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    #mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)

    cdict2 = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.1),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
              }

    blue_red2 = LinearSegmentedColormap('BlueRed2', cdict2)
    pyplot.register_cmap(cmap=blue_red2)
    #pyplot.rcParams['image.cmap'] = 'BlueRed2'

    colormap = pyplot.get_cmap('BlueRed2')

    #ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    #cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=matplotlib.cm.coolwarm,
    #                            norm=norm,
    #                            orientation='vertical')

    times = numpy.arange(0.0, 10.05, 0.05)

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        disked_stars = small_stars[small_stars.dispersed == False]

        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]

        # Open final files
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, 10.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        final_stars = []
        for s in stars:
            if s in disked_stars:
                final_stars
        final_mass_loss_trunc = final_stars.truncation_mass_loss.value_in(units.MSun)
        final_mass_loss_photoevap = final.photoevap_mass_loss.value_in(units.MSun)

        if t == 0.0:  # Have to do this here because of a mess up when saving... will not be needed for future results
            mass_loss_trunc = numpy.zeros((len(disked_stars)))
            mass_loss_photoevap = numpy.zeros((len(disked_stars)))
        else:
            mass_loss_trunc = disked_stars.truncation_mass_loss.value_in(units.MSun)
            mass_loss_photoevap = disked_stars.photoevap_mass_loss.value_in(units.MSun)

        zero = numpy.zeros((len(mass_loss_trunc)))
        plot_mass_loss_map = (zero - mass_loss_trunc + mass_loss_photoevap) / (final_mass_loss_photoevap +final_mass_loss_trunc)
        #print plot_mass_loss_map[:10]

        pyplot.clf()
        ax = pyplot.gca()
        ax.set_title('N={0}, c=100, t={1:.2f} Myr'.format(N, t))
        ax.set_xlabel('x [parsec]')
        ax.set_ylabel('y [parsec]')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Disks
        ml = ax.scatter(disked_stars.x.value_in(units.parsec), disked_stars.y.value_in(units.parsec),
                        s=disked_stars.disk_radius.value_in(units.au), #norm=norm,
                        c=plot_mass_loss_map, cmap=colormap, alpha=0.5, vmin=-1, vmax=1)

        # Bright stars
        ax.scatter(bright_stars.x.value_in(units.parsec), bright_stars.y.value_in(units.parsec),
                   marker='*', color='k', s=8*bright_stars.stellar_mass.value_in(units.MSun))

        #ax.legend(loc='upper right', fontsize=20)
        pyplot.colorbar(mappable=ml)
        #pyplot.show()
        #fig.savefig('{0}/plot{1:.2f}.png'.format(save_path, t))
        fig.savefig('{0}/plot{1}.png'.format(save_path, numpy.where(times==t)[0][0]))


def size_in_time(open_path, save_path, N, i):  # For one star
    #fig = pyplot.figure(figsize=(12, 8))
    #ax = pyplot.gca()
    fig, (ax, ax2) = pyplot.subplots(2, 1, figsize=(12, 10))#, gridspec_kw={"width_ratios": [12, 1]})
    ax.set_title('Distance to nearest bright stars')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Distance [parsec]')

    ax2.set_title('Disk size')
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Disk size [au]')
    #ax.set_xlim(-1.5, 1.5)
    #ax.set_ylim(-1.5, 1.5)
    #ax.axis('square')
    #pyplot.axes(ax).set_aspect('equal', adjustable='box')

    #ax2 = ax.twinx()

    times = numpy.arange(0.0, 10.05, 0.05)

    disk_size, distance_to_bright = [], []

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        s = stars[i]

        distances = []

        for bs in bright_stars:
            distances.append(distance(bs, s).value_in(units.parsec))

        nearest_bs = min(distances)

        distance_to_bright.append(nearest_bs)
        disk_size.append(2 * s.disk_radius.value_in(units.au))

    disk_size[1] = disk_size[0]  # this has to do with the saving issue, will fix soon
    disk_size[2] = disk_size[0]

    ax.plot(times, distance_to_bright)
    ax2.plot(times, disk_size)

    #ax.legend(loc='upper left', fontsize=20)

    pyplot.tight_layout()
    #pyplot.show()


def mass_loss_in_time(open_path, save_path, N, i):  # For one star
    #fig = pyplot.figure(figsize=(12, 8))
    #ax = pyplot.gca()
    fig, (ax, ax2) = pyplot.subplots(2, 1, figsize=(12, 10))#, gridspec_kw={"width_ratios": [12, 1]})
    ax.set_title('Mass loss in time')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Mass loss [MSun]')

    ax2.set_title('Distance to nearest bright stars')
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Distance [parsec]')

    #ax.set_xlim(-1.5, 1.5)
    #ax.set_ylim(-1.5, 1.5)
    #ax.axis('square')
    #pyplot.axes(ax).set_aspect('equal', adjustable='box')

    #ax2 = ax.twinx()

    times = numpy.arange(0.0, 10.05, 0.05)

    photoevap_mass_loss, dyn_trunc_mass_loss, disk_size, distance_to_bright = [], [], [], []

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        s = stars[i]

        photoevap_mass_loss.append(s.photoevap_mass_loss.value_in(units.MSun))
        dyn_trunc_mass_loss.append(s.truncation_mass_loss.value_in(units.MSun))
        disk_size.append(2 * s.disk_radius.value_in(units.au))

    disk_size[1] = disk_size[0]  # this has to do with the saving issue, will fix soon
    disk_size[2] = disk_size[0]

    ax.plot(times, photoevap_mass_loss, c='r', label="photoevap")
    ax.plot(times, dyn_trunc_mass_loss, c='b', label="dyn trunc")
    ax2.plot(times, disk_size, c='k')

    ax.legend(loc='upper left', fontsize=20)

    pyplot.tight_layout()
    #pyplot.show()


def single_star(open_path, save_path, N, i, t_end, all_distances=0):
    #fig = pyplot.figure(figsize=(12, 8))
    #ax = pyplot.gca()
    fig, axs = pyplot.subplots(2, 2, figsize=(18, 14))

    ax = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # Mass loss
    ax.set_title('Mass loss in time')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'Mass loss [$M_J$]')
    ax.set_xlim(0, t_end)

    # Distance
    ax2.set_title('Distance to nearest bright star')
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Distance [parsec]')

    # Disk size
    ax3.set_title('Disk size')
    ax3.set_xlabel('Time [Myr]')
    ax3.set_ylabel('Disk size [au]')
    #ax3.set_xlim([0, t_end])

    # G0
    ax4.set_title(r'$G_0$')
    ax4.set_xlabel('Time [Myr]')
    ax4.set_ylabel(r'$G_0$')

    #ax.set_xlim(-1.5, 1.5)
    #ax.set_ylim(-1.5, 1.5)
    #ax.axis('square')
    #pyplot.axes(ax).set_aspect('equal', adjustable='box')

    #ax2 = ax.twinx()

    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    disk_sizes = []
    initial_size = 0.0

    mass_loss_pe, mass_loss_trunc = [], []
    g0s = []

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        bright_stars = stars[stars.stellar_mass.value_in(units.MSun) > 1.9]
        s = stars[i]
        if t == 0.0:
            initial_size = s.disk_radius.value_in(units.au)
        fig.suptitle(r"$M_*$ = {0:.2f} $M_\odot$, $R_d$ = {1:.2f} au, $M_d$ = {2:.2f} $M_J$".format(s.stellar_mass.value_in(units.MSun), s.initial_disk_size.value_in(units.au), s.initial_disk_mass.value_in(units.MJupiter)))

        if not s.dispersed:
            disk_sizes.append(s.disk_radius.value_in(units.au))
            mass_loss_pe.append(s.photoevap_mass_loss.value_in(units.MSun))
            mass_loss_trunc.append(s.truncation_mass_loss.value_in(units.MSun))
        else:
            disk_sizes.append(0.0)

        g0s.append(s.g0)

    mass_loss_pe = numpy.pad(mass_loss_pe, (0, len(times)-len(mass_loss_pe)), 'edge')
    mass_loss_trunc = numpy.pad(mass_loss_trunc, (0, len(times)-len(mass_loss_trunc)), 'edge')

    ax3.plot(times, disk_sizes)
    ax3.axhline(initial_size, ls=":")
    ax3.text(1.0, 7*initial_size/8, "Init radius = {0} au".format(initial_size))

    ax.plot(times, mass_loss_pe, color="red")
    ax.plot(times, mass_loss_trunc, color="blue")

    ax4.plot(times, g0s)

    pyplot.show()


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


def cdfs_w_old(open_path, labels, colors, density, N, t):
    """ Plot cumulative distributions of disk sizes (au).

    :param files: list with filenames to open
    :param labels: list with labels for each file
    :param colors: list with colors for each file
    """
    fig = pyplot.figure(figsize=(10, 8), dpi=90)
    ax = pyplot.subplot(111)

    gas_scenarios = ['no_gas']#, 'gas', 'gas_exp'] # 'medium_exp', 'late_exp']
    plot_labels = ['No gas']  # 'Medium gas expulsion', 'Late gas expulsion']

    #	alphas = [0.0001, 0.002, 0.005, 0.007, 0.01]
    alphas = [0.005]
    # plot_labels = [r'$\alpha = {10}^{-4}$', r'$\alpha = 2 \times {10}^{-3}$', r'$\alpha = 5 \times {10}^{-3}$', r'$\alpha = 7 \times {10}^{-3}$', r'$\alpha = {10}^{-2}$']
    line_styles = ['-']
    Rc = 30.0

    path = '/media/fran/data1/Trapezium/redo_params2/alpha'
    snapshot = 2000000
    Rvir = 0.5
    runs = 5
    N = 1500

    limit = 250

    for alpha, line in zip(alphas, line_styles):
        for scenario, label in zip(gas_scenarios, plot_labels):
            all_runs = []
            for r in range(runs):
                file_path = '{0}/{1}/{2}/{3}/{4}/{5}/R{6}_{7}.hdf5'.format(path, scenario, N, alpha, Rc, r, Rvir,
                                                                           snapshot)
                # file_path = '/data2/Trapezium/{0}/{1}/{2}/{3}/{4}/{5}.hdf5'.format(path, scenario, N, alpha, r, snapshot)
                stars = io.read_set_from_file(file_path, 'hdf5')
                disk_size = (2 * stars.initial_characteristic_disk_radius.value_in(units.AU))
                # disk_size = 2 * stars.initial_characteristic_disk_radius.value_in(units.AU)
                sorted_disk_size = numpy.sort(disk_size)
                all_runs.append(sorted_disk_size)

            all_runs = numpy.array(all_runs)
            sorted_disk_sizes = numpy.mean(all_runs, axis=0)
            small_dyn = len(sorted_disk_sizes[sorted_disk_sizes < limit])
            all_dyn = len(sorted_disk_sizes)
            print small_dyn, all_dyn
            sorted_disk_size_errors = numpy.std(all_runs, axis=0)
            cumulative = numpy.array([float(x) for x in numpy.arange(sorted_disk_sizes.size + 1)])

            ax.plot(numpy.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]), cumulative / len(cumulative),
                     ls=line, lw=3, c=colors[1], label="Dynamical truncations only")

    #pyplot.xlim([1.0, 3.5])
    pyplot.ylim([0.0, 1.0])
    # pyplot.legend(loc='upper left', numpoints=1, fontsize=18)

    N = 100

    for p, l in zip(open_path, labels):
        f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
        print "cdfs opening", f
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        disked_stars2 = small_stars[small_stars.dispersed == False]
        disked_stars = disked_stars2[disked_stars2.photoevap_mass_loss < disked_stars2.initial_disk_mass]

        print len(disked_stars)

        sizes, masses = (2 * disked_stars.disk_radius.value_in(units.au)), disked_stars.disk_mass.value_in(units.MJupiter)

        sorted_disk_sizes = numpy.sort(sizes)
        sorted_disk_masses = numpy.sort(masses)

        small_phot = len(sorted_disk_sizes[sorted_disk_sizes < limit])
        all_phot = len(sorted_disk_sizes)
        print small_phot, all_phot

        cumulative_sizes = numpy.array([float(x) for x in numpy.arange(sorted_disk_sizes.size + 1)])
        cumulative_masses = numpy.array([float(x) for x in numpy.arange(sorted_disk_masses.size + 1)])

        ax.plot(numpy.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]),
                 cumulative_sizes / len(cumulative_sizes), c=colors[0],
                 lw=3, label="Dynamical truncations\nand ext. photoevaporation")

    dyn_percent = 100. * (float(small_dyn)/all_dyn)
    pe_percent = 100. * (float(small_phot)/all_phot)
    ax.annotate('{0:.3}\% disks under 250 au'.format(dyn_percent), xy=(500, 0.5), fontsize=40, color=colors[1])
    ax.annotate('{0:.3}\% disks under 250 au'.format(pe_percent), xy=(500, 0.4), fontsize=40, color=colors[0])

    ax.axvline(limit, c="k", ls=":")
    ax.legend(loc='lower right', fontsize=20)

    ax.set_title('CDF of disk sizes, t = {0} Myr'.format(t))
    ax.set_xlabel(r'$d_{disk}$ [au]')
    ax.set_ylabel(r'$f < d_{disk}$')

    pyplot.tight_layout()
    pyplot.show()
    #fig.savefig('plot2.png')

def cdfs_in_time(p, save_path, N, times):
    """ Plot cumulative distributions of disk sizes (au) and masses (MJup).

    :param open_path: list of folders to use
    :param save_path: location where the figures are saved
    :param N: number of stars
    :param t: time to show in plot
    """
    pyplot.figure(1)
    pyplot.figure(2)


    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        #density = 1E-7
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        #small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
         #                         (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        small_stars = small_stars[small_stars.dispersed == False]
        #disked_stars = small_stars[small_stars.photoevap_mass_loss < small_stars.initial_disk_mass]

        disk_sizes, disk_masses = 2. * small_stars.disk_radius.value_in(units.au), small_stars.disk_mass.value_in(units.MJupiter)

        sorted_disk_sizes = numpy.sort(disk_sizes)
        sorted_disk_masses = numpy.sort(disk_masses)

        cumulative_sizes = numpy.array([float(x) for x in numpy.arange(sorted_disk_sizes.size + 1)])
        cumulative_masses = numpy.array([float(x) for x in numpy.arange(sorted_disk_masses.size + 1)])

        pyplot.figure(1)
        pyplot.plot(numpy.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]),
                cumulative_sizes / len(cumulative_sizes),
                lw=2, label=t)

        pyplot.figure(2)
        pyplot.plot(numpy.concatenate([sorted_disk_masses, sorted_disk_masses[[-1]]]),
                cumulative_masses / len(cumulative_masses),
                lw=2, label=t)

    pyplot.figure(1)
    ax1 = pyplot.gca()
    ax1.legend()
    #ax1.legend(loc='lower right', fontsize=20)
    #ax1.set_title('CDF of disk sizes, t = {0} Myr'.format(t))
    ax1.set_xlabel(r'$d_{disk}$ [au]')
    ax1.set_ylabel(r'$f < d_{disk}$')
    ax1.annotate('{0} surviving disks'.format(len(disk_sizes)), xy=(300, 0.2), fontsize=40)
    pyplot.tight_layout()
    pyplot.savefig('{0}/CDF_size_in_time.png'.format(save_path, t))

    pyplot.figure(2)
    ax2 = pyplot.gca()
    ax2.legend()
    #ax2.legend(loc='lower right', fontsize=20)
    #ax2.set_title('CDF of disk masses, t = {0} Myr'.format(t))
    ax2.set_xlabel(r'$M_{disk}$ [$M_{Jupiter}$]')
    ax2.set_ylabel(r'$f < M_{disk}$')
    ax2.annotate('{0} surviving disks'.format(len(disk_sizes)), xy=(30, 0.2), fontsize=40)
    pyplot.tight_layout()
    pyplot.savefig('{0}/CDF_mass_in_time.png'.format(save_path, t))

    pyplot.show()



def cdfs(open_path, save_path, N, t):
    """ Plot cumulative distributions of disk sizes (au) and masses (MJup).

    :param open_path: list of folders to use
    :param save_path: location where the figures are saved
    :param N: number of stars
    :param t: time to show in plot
    """
    pyplot.figure(1)
    pyplot.figure(2)

    all_sorted_disk_masses, all_sorted_disk_sizes = [], []

    for p in open_path:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        # Take only the small stars
        density = 1E-7
        small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
        small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
                                  (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
        #disked_stars2 = small_stars[small_stars.dispersed == False]
        disked_stars = small_stars[small_stars.photoevap_mass_loss < small_stars.initial_disk_mass]

        sizes, masses = 2. * disked_stars.disk_radius.value_in(units.au), disked_stars.disk_mass.value_in(units.MJupiter)

        sorted_disk_sizes = numpy.sort(sizes)
        sorted_disk_masses = numpy.sort(masses)

        all_sorted_disk_sizes.append(sorted_disk_sizes)
        all_sorted_disk_masses.append(sorted_disk_masses)

    all_sorted_disk_sizes[0] = numpy.array([0.] + list(all_sorted_disk_sizes[0]))
    all_sorted_disk_masses[0] = numpy.array([0.] + list(all_sorted_disk_masses[0]))

    print all_sorted_disk_sizes

    disk_sizes = numpy.mean(all_sorted_disk_sizes, axis=0)
    disk_masses = numpy.mean(all_sorted_disk_masses, axis=0)

    disk_sizes_stdev = numpy.std(all_sorted_disk_sizes, axis=0)
    disk_masses_stdev = numpy.std(all_sorted_disk_masses, axis=0)

    cumulative_sizes = numpy.array([float(x) for x in numpy.arange(disk_sizes.size + 1)])
    cumulative_masses = numpy.array([float(x) for x in numpy.arange(disk_masses.size + 1)])

    sizes_low = numpy.concatenate([disk_sizes, disk_sizes[[-1]]]) \
          - numpy.concatenate([disk_sizes_stdev, disk_sizes_stdev[[-1]]])
    sizes_high = numpy.concatenate([disk_sizes, disk_sizes[[-1]]]) \
           + numpy.concatenate([disk_sizes_stdev, disk_sizes_stdev[[-1]]])

    masses_low = numpy.concatenate([disk_masses, disk_masses[[-1]]]) \
          - numpy.concatenate([disk_masses_stdev, disk_masses_stdev[[-1]]])
    masses_high = numpy.concatenate([disk_masses, disk_masses[[-1]]]) \
           + numpy.concatenate([disk_masses_stdev, disk_masses_stdev[[-1]]])

    pyplot.figure(1)
    pyplot.plot(numpy.concatenate([disk_sizes, disk_sizes[[-1]]]),
                cumulative_sizes / len(cumulative_sizes),
                lw=2)
    pyplot.fill_betweenx(cumulative_sizes / len(cumulative_sizes),
                         sizes_low, sizes_high,
                         alpha='0.2')

    pyplot.figure(2)
    pyplot.plot(numpy.concatenate([disk_masses, disk_masses[[-1]]]),
                cumulative_masses / len(cumulative_masses),
                lw=2)
    pyplot.fill_betweenx(cumulative_masses / len(cumulative_masses),
                         masses_low, masses_high,
                         alpha='0.2')

    pyplot.figure(1)
    ax1 = pyplot.gca()
    #ax1.legend(loc='lower right', fontsize=20)
    #ax1.set_title('CDF of disk sizes, t = {0} Myr'.format(t))
    ax1.set_xlabel(r'$d_{disk}$ [au]')
    ax1.set_ylabel(r'$f < d_{disk}$')
    ax1.annotate('{0} surviving disks'.format(len(disk_sizes)), xy=(300, 0.2), fontsize=40)
    pyplot.tight_layout()
    pyplot.savefig('{0}/CDF_size_t{1}.png'.format(save_path, t))

    pyplot.figure(2)
    ax2 = pyplot.gca()
    #ax2.legend(loc='lower right', fontsize=20)
    #ax2.set_title('CDF of disk masses, t = {0} Myr'.format(t))
    ax2.set_xlabel(r'$M_{disk}$ [$M_{Jupiter}$]')
    ax2.set_ylabel(r'$f < M_{disk}$')
    ax2.annotate('{0} surviving disks'.format(len(disk_sizes)), xy=(30, 0.2), fontsize=40)
    pyplot.tight_layout()
    pyplot.savefig('{0}/CDF_mass_t{1}.png'.format(save_path, t))

    pyplot.show()


def cdfs_with_observations_size(open_path, save_path, N, times, colors, labels, log=False):
    """ Plot cumulative distributions of disk sizes (au) and masses (MJup).

    :param open_path: list of folders to use
    :param save_path: location where the figures are saved
    :param N: number of stars
    :param t: time to show in plot
    :param log: if True, plot in logscale
    """
    fig = pyplot.figure()
    axs00 = pyplot.subplot2grid((2, 6), (0, 0), colspan=2)
    axs01 = pyplot.subplot2grid((2, 6), (0, 2), colspan=2)
    axs02 = pyplot.subplot2grid((2, 6), (0, 4), colspan=2)
    axs10 = pyplot.subplot2grid((2, 6), (1, 1), colspan=2)
    axs11 = pyplot.subplot2grid((2, 6), (1, 3), colspan=2)

    for t in times:
        all_sorted_disk_sizes = []
        for p in open_path:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            # Take only the small stars
            small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
            small_stars = small_stars[small_stars.dispersed == False]
            #print small_stars.disk_mass.value_in(units.g) / (math.pi * small_stars.disk_radius.value_in(units.cm)**2)
            small_stars = small_stars[1. < small_stars.disk_mass.value_in(units.g) / (math.pi * small_stars.disk_radius.value_in(units.cm)**2)]

            if log:
                sizes = 2. * numpy.log10(small_stars.disk_radius.value_in(units.au))
            else:
                sizes = 2. * small_stars.disk_radius.value_in(units.au)

            sorted_disk_sizes = numpy.sort(sizes)
            all_sorted_disk_sizes.append(sorted_disk_sizes)

        try:
            disk_sizes = numpy.mean(all_sorted_disk_sizes, axis=0)
            disk_sizes_stdev = numpy.std(all_sorted_disk_sizes, axis=0)
        except ValueError:
            max_len = 0
            for a in all_sorted_disk_sizes:
                if len(a) > max_len:
                    max_len = len(a)

            new_sorted = []
            for a in all_sorted_disk_sizes:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant')
                new_sorted.append(b)

            disk_sizes = numpy.mean(new_sorted, axis=0)
            disk_sizes_stdev = numpy.std(new_sorted, axis=0)

        cumulative_sizes = 1. * numpy.arange(len(disk_sizes)) / (len(disk_sizes) - 1)

        sizes_low = disk_sizes - disk_sizes_stdev
        sizes_high = disk_sizes + disk_sizes_stdev

        # Plotting together with observational data now, according to their age t

        # For plots
        xtext = 180
        ytext = 0.05
        xlimits = [0, 500]
        ylimits = [0.0, 1.0]
        ticks = [0, 250, 500]
        xlabel = '$d_{disk}$ [au]'
        ylabel = '$f < d_{disk}$'

        if t == 1.:
            axs00.plot(disk_sizes,
                       cumulative_sizes,
                       lw=2, color='black')
            axs00.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs00.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs00.set_xlim(xlimits)
            axs00.set_ylim(ylimits)
            axs00.set_xticks(ticks)

            """# Trapezium data (Vicente & Alves 2005)
            lines = open('data/Trapezium_sizes.txt', 'r').readlines()
            trapezium_sizes = []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split()[8]
                trapezium_sizes.append(float(data))

            trapezium_sizes = numpy.array(trapezium_sizes)

            if log:
                sorted_trapezium_sizes = numpy.sort(numpy.log10(trapezium_sizes[trapezium_sizes > 100.]))
            else:
                sorted_trapezium_sizes = numpy.sort(trapezium_sizes[trapezium_sizes > 100.])

            p = 1. * numpy.arange(len(sorted_trapezium_sizes)) / (len(sorted_trapezium_sizes) - 1)

            axs00.plot(sorted_trapezium_sizes,
                           p,
                           ls='-', lw=2,
                           color=colors[0],
                           label=labels[0])
            axs00.set_title('Trapezium')
            #axs00.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)"""

            # ONC data (Eisner+ 2018)
            lines = open('data/ONC.txt', 'r').readlines()
            onc_sizes, onc_sizes_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split('&')[7]
                b = data.split('$')[1]
                c = b.split('\pm')
                if len(c) == 2:  # Value + error
                    onc_sizes.append(2. * float(c[0]))  # 2. factor for radius to diameter
                    onc_sizes_error.append(2. * float(c[1]))
                else:  # Upper limit
                    d = c[0].split('<')[1][1:]
                    onc_sizes.append(float(d))
                    onc_sizes_error.append(0.0)

            if log:
                sorted_onc_sizes = numpy.sort(numpy.log10(onc_sizes))
                sorted_onc_sizes_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(onc_sizes, onc_sizes_error))])
            else:
                sorted_onc_sizes = numpy.sort(onc_sizes)
                sorted_onc_sizes_errors = numpy.array([x for _, x in sorted(zip(onc_sizes, onc_sizes_error))])

            p = 1. * numpy.arange(len(sorted_onc_sizes)) / (len(sorted_onc_sizes) - 1)

            onc_low = sorted_onc_sizes - sorted_onc_sizes_errors
            onc_high = sorted_onc_sizes + sorted_onc_sizes_errors

            axs00.plot(sorted_onc_sizes, p,
                           ls='-', lw=2,
                           color=colors[0],
                           label=labels[0])
            axs00.fill_betweenx(p,
                                onc_low, onc_high,
                                alpha='0.2', facecolor=colors[0])
            axs00.set_title('ONC')
            #axs00.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)

        elif t == 2.:
            axs01.plot(disk_sizes,
                       cumulative_sizes,
                        lw=2, color='black')
            axs01.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs01.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs01.set_xlim(xlimits)
            axs01.set_ylim(ylimits)
            axs01.set_xticks(ticks)

            # Lupus data (Ansdell et al 2018)
            gas_disk_sizes, gas_disk_sizes_error = [], []

            lines = open('data/Lupus_sizes.txt', 'r').readlines()
            for line in (line for line in lines if not line.startswith('#')):
                r_gas = line.split()[5]
                r_gas_error = line.split()[6]
                gas_disk_sizes.append(float(r_gas))
                gas_disk_sizes_error.append(float(r_gas_error))

            if log:
                sorted_lupus_disk_sizes = numpy.sort(numpy.log10(gas_disk_sizes))
                sorted_lupus_disk_sizes_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(gas_disk_sizes, gas_disk_sizes_error))])

            else:
                sorted_lupus_disk_sizes = numpy.sort(gas_disk_sizes)
                sorted_lupus_disk_sizes_errors = numpy.array([x for _, x in sorted(zip(gas_disk_sizes, gas_disk_sizes_error))])

            p = 1. * numpy.arange(len(sorted_lupus_disk_sizes)) / (len(sorted_lupus_disk_sizes) - 1)

            low_lupus = sorted_lupus_disk_sizes - sorted_lupus_disk_sizes_errors
            high_lupus = sorted_lupus_disk_sizes + sorted_lupus_disk_sizes_errors

            axs01.plot(sorted_lupus_disk_sizes, p,
                           ls='-', lw=2,
                           color=colors[1],
                           label=labels[1])
            axs01.fill_betweenx(p,
                                    low_lupus, high_lupus,
                                    alpha='0.2', facecolor=colors[1])
            axs01.set_title('Lupus clouds')
            #axs01.legend()
            axs01.set_xlabel(xlabel)
            axs01.set_ylabel(ylabel)

        elif t == 2.5:
            axs02.plot(disk_sizes,
                       cumulative_sizes,
                        lw=2, color='black')
            axs02.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs02.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs02.set_xlim(xlimits)
            axs02.set_ylim(ylimits)
            axs02.set_xticks(ticks)

            # Chamaeleon I data (Pascucci et al 2016)
            lines = open('data/ChamI_sizes.txt', 'r').readlines()
            cham_sizes_arsec = []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[7]
                b = line.split()[8]
                if a > b:
                    cham_sizes_arsec.append(float(a))
                else:
                    cham_sizes_arsec.append(float(b))

            cham_sizes_arsec = numpy.array(cham_sizes_arsec)
            cham_sizes_arsec = cham_sizes_arsec[cham_sizes_arsec > 0.0]

            cham_distance_pc = 160
            cham_distance_au = 2.0626 * pow(10, 5) * cham_distance_pc
            cham_sizes_au = (numpy.pi / 180) * (cham_sizes_arsec / 3600.) * cham_distance_au

            if log:
                cham_sorted_disk_sizes = numpy.sort(numpy.log10(cham_sizes_au))
            else:
                cham_sorted_disk_sizes = numpy.sort(cham_sizes_au)

            p = 1. * numpy.arange(len(cham_sorted_disk_sizes)) / (len(cham_sorted_disk_sizes) - 1)

            axs02.plot(cham_sorted_disk_sizes, p,
                           ls='-', lw=2,
                           color=colors[2],
                           label=labels[2])
            axs02.set_title('Chamaeleon I')
            #axs02.legend()
            axs02.set_xlabel(xlabel)
            axs02.set_ylabel(ylabel)
            axs02.set_xticks([0, 250, 500])

        elif t == 4.:
            axs10.plot(disk_sizes,
                       cumulative_sizes,
                        lw=2, color='black')
            axs10.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs10.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs10.set_xlim(xlimits)
            axs10.set_ylim(ylimits)
            axs10.set_xticks(ticks)

            # sigma Orionis data (Mauco et al 2016)
            lines = open('data/sigmaOrionis_sizes.txt', 'r').readlines()
            sOrionis_sizes_au, sOrionis_sizes_low, sOrionis_sizes_high = [], [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[1]
                b = line.split()[2][1:-1]
                c, d = b.split('-')
                sOrionis_sizes_au.append(float(a))
                sOrionis_sizes_low.append(float(c))
                sOrionis_sizes_high.append(float(d))

            if log:
                sOrionis_sorted_disk_sizes = numpy.sort(numpy.array(numpy.log10(sOrionis_sizes_au)))
                sOrionis_sorted_low = numpy.array([numpy.log10(x) for _, x in sorted(zip(sOrionis_sizes_au, sOrionis_sizes_low))])
                sOrionis_sorted_high = numpy.array([numpy.log10(x) for _, x in sorted(zip(sOrionis_sizes_au, sOrionis_sizes_high))])

            else:
                sOrionis_sorted_disk_sizes = numpy.sort(numpy.array(sOrionis_sizes_au))
                sOrionis_sorted_low = numpy.array([x for _, x in sorted(zip(sOrionis_sizes_au, sOrionis_sizes_low))])
                sOrionis_sorted_high = numpy.array([x for _, x in sorted(zip(sOrionis_sizes_au, sOrionis_sizes_high))])

            p = 1. * numpy.arange(len(sOrionis_sorted_disk_sizes)) / (len(sOrionis_sorted_disk_sizes) - 1)

            axs10.plot(sOrionis_sorted_disk_sizes, p,
                           ls='-', lw=2,
                           color=colors[3],
                           label=labels[3])
            axs10.fill_betweenx(p,
                                    sOrionis_sorted_low, sOrionis_sorted_high,
                                    alpha='0.2', facecolor=colors[3])
            axs10.set_title('$\sigma$ Orionis')
            #axs10.legend()
            axs10.set_xlabel(xlabel)
            axs10.set_ylabel(ylabel)

        elif t == 5.:
            axs11.plot(disk_sizes,
                       cumulative_sizes,
                        lw=2, color='black')
            axs11.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs11.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs11.set_xlim(xlimits)
            axs11.set_ylim(ylimits)
            axs11.set_xticks(ticks)

            # UpperSco data (Barenfeld et al 2017)
            lines = open('data/UpperSco_sizes.txt', 'r').readlines()
            uppersco_sizes, uppersco_errors_low, uppersco_errors_high = [], [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[7]
                b = line.split()[8]
                c = line.split()[9]
                uppersco_sizes.append(float(a))
                uppersco_errors_low.append(float(b[2:-1]))
                uppersco_errors_high.append(float(c[1:-1]))

            if log:
                uppersco_sorted_disk_sizes = numpy.sort(numpy.log10(uppersco_sizes))
                uppersco_low_sorted = numpy.array([numpy.log10(x) for _, x in sorted(zip(uppersco_sizes, uppersco_errors_low))])
                uppersco_high_sorted = numpy.array([numpy.log10(x) for _, x in sorted(zip(uppersco_sizes, uppersco_errors_high))])
            else:
                uppersco_sorted_disk_sizes = numpy.sort(uppersco_sizes)
                uppersco_low_sorted = numpy.array([x for _, x in sorted(zip(uppersco_sizes, uppersco_errors_low))])
                uppersco_high_sorted = numpy.array([x for _, x in sorted(zip(uppersco_sizes, uppersco_errors_high))])

            uppersco_low = uppersco_sorted_disk_sizes - uppersco_low_sorted
            uppersco_high = uppersco_sorted_disk_sizes + uppersco_high_sorted

            p = 1. * numpy.arange(len(uppersco_sorted_disk_sizes)) / (len(uppersco_sorted_disk_sizes) - 1)

            axs11.plot(uppersco_sorted_disk_sizes, p,
                           ls='-', lw=2,
                           color=colors[4],
                           label=labels[4])
            axs11.fill_betweenx(p,
                                    uppersco_low, uppersco_high,
                                    alpha='0.2', facecolor=colors[4])
            axs11.set_title('UpperSco')
            #axs11.legend()
            axs11.set_xlabel(xlabel)
            axs11.set_ylabel(ylabel)

    pyplot.tight_layout()
    pyplot.savefig('{0}/CDF_data_size.png'.format(save_path))

    pyplot.show()


def cdfs_with_observations_mass(open_path, save_path, N, times, colors, labels, log=False):
    """ Plot cumulative distributions of disk sizes (au) and masses (MJup).

    :param open_path: list of folders to use
    :param save_path: location where the figures are saved
    :param N: number of stars
    :param t: time to show in plot
    :param log: if True, plot in logscale
    """
    # Changed the order of the subplots to make sure the plots are in chronological order
    fig = pyplot.figure(figsize=(14, 12))
    axs00 = pyplot.subplot2grid((6, 6), (0, 0), colspan=2, rowspan=2)  # ONC, 1 Myr
    axs01 = pyplot.subplot2grid((6, 6), (2, 0), colspan=2, rowspan=2)  # Lupus, 2 Myr
    axs02 = pyplot.subplot2grid((6, 6), (2, 2), colspan=2, rowspan=2)  # ChamI, 2.5 Myr
    axs10 = pyplot.subplot2grid((6, 6), (4, 0), colspan=2, rowspan=2)  # sOri, 4 Myr
    axs11 = pyplot.subplot2grid((6, 6), (4, 2), colspan=2, rowspan=2)  # Upper Sco, 5 Myr
    axs12 = pyplot.subplot2grid((6, 6), (2, 4), colspan=2, rowspan=2)  # IC348, 2.5 Myr
    axs20 = pyplot.subplot2grid((6, 6), (0, 4), colspan=2, rowspan=2)  # OMC-2, 1 Myr
    axs21 = pyplot.subplot2grid((6, 6), (0, 2), colspan=2, rowspan=2)  # OMC-1, 1 Myr

    for t in times:
        all_sorted_disk_masses = []
        for p in open_path:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            # Take only the small stars
            small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
            small_stars = small_stars[small_stars.dispersed == False]

            if log:
                disk_masses = numpy.log10(small_stars.disk_mass.value_in(units.MJupiter))
            else:
                disk_masses = small_stars.disk_mass.value_in(units.MJupiter)

            sorted_disk_masses = numpy.sort(disk_masses)
            all_sorted_disk_masses.append(sorted_disk_masses)

        try:
            disk_masses = numpy.median(all_sorted_disk_masses, axis=0)
            disk_masses_stdev = numpy.std(all_sorted_disk_masses, axis=0)
        except ValueError:
            max_len = 0
            for a in all_sorted_disk_masses:
                if len(a) > max_len:
                    max_len = len(a)

            new_sorted = []
            for a in all_sorted_disk_masses:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant')
                new_sorted.append(b)

            disk_masses = numpy.median(new_sorted, axis=0)
            disk_masses_stdev = numpy.std(new_sorted, axis=0)

        cumulative_masses = 1. * numpy.arange(len(disk_masses)) / (len(disk_masses) - 1)

        masses_low = disk_masses - disk_masses_stdev
        masses_high = disk_masses + disk_masses_stdev

        # For plots
        xlimits = [-2.5, 3.5]
        ylimits = [0.0, 1.0]
        ticks = [-2, 0, 2]
        yticks = [0.0, 0.5, 1.0]
        xtext = 0.7
        ytext = 0.05
        xlabel = '$\log(M_{disk})$ [$M_{Jupiter}$]'
        ylabel = '$f < M_{disk}$'

        if t == 1.0:
            axs00.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs00.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs00.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs00.set_xlim(xlimits)
            axs00.set_ylim(ylimits)
            axs00.set_xticks(ticks)
            axs00.set_yticks(yticks)

            axs20.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs20.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs20.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs20.set_xlim(xlimits)
            axs20.set_ylim(ylimits)
            axs20.set_xticks(ticks)
            axs20.set_yticks(yticks)

            axs21.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs21.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs21.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs21.set_xlim(xlimits)
            axs21.set_ylim(ylimits)
            axs21.set_xticks(ticks)
            axs21.set_yticks(yticks)


            """# Trapezium data (Mann & Williams 2009 2009ApJ...694L..36M)
            # Data: dust masses in 1E-2 MSun
            r_path = 'data/Trapezium_masses.txt'
            lines = open(r_path, 'r').readlines()

            trapezium_masses = []
            trapezium_masses_errors = []

            for line in (l for l in lines if not l.startswith('#')):
                a = line.split()[7]
                b = line.split()[9]

                # Unit conversion: 1E-2 MSun to MJup
                ms = float(a) * 1E-2 | units.MSun
                mj = ms.value_in(units.MJupiter)

                ms_error = float(b) * 1E-2 | units.MSun
                mj_error = ms_error.value_in(units.MJupiter)

                trapezium_masses.append(mj)
                trapezium_masses_errors.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            trapezium_masses = 100. * numpy.asarray(trapezium_masses)
            trapezium_masses_errors = 100. * numpy.asarray(trapezium_masses_errors)

            if log:
                trapezium_sorted_masses = numpy.sort(numpy.log10(trapezium_masses))
                trapezium_sorted_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(trapezium_masses, trapezium_masses_errors))])
            else:
                trapezium_sorted_masses = numpy.sort(trapezium_masses)
                trapezium_sorted_errors = numpy.array([x for _, x in sorted(zip(trapezium_masses, trapezium_masses_errors))])

            p = 1. * numpy.arange(len(trapezium_sorted_masses)) / (len(trapezium_sorted_masses) - 1)
            trapezium_low = trapezium_sorted_masses - trapezium_sorted_errors
            trapezium_high = trapezium_sorted_masses + trapezium_sorted_errors

            axs00.plot(trapezium_sorted_masses, p,
                           ls='-', lw=3,
                           color=colors[0],
                           label=labels[0])
            axs00.fill_betweenx(p,
                                    trapezium_low, trapezium_high,
                                    alpha='0.2', facecolor=colors[0])
            axs00.set_title('Trapezium')
            #axs00.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)"""

            # ONC data (Eisner+ 2018)
            # Data: DUST masses in MEarth
            lines = open('data/ONC.txt', 'r').readlines()
            onc_masses, onc_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split('&')[6]
                a = data.split('$\pm$')[0]
                b = data.split('$\pm$')[1]

                # Unit conversion
                me = float(a) | units.MEarth
                mj = me.value_in(units.MJupiter)

                me_error = float(b) | units.MEarth
                mj_error = me_error.value_in(units.MJupiter)

                onc_masses.append(mj)
                onc_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            onc_masses = 100. * numpy.asarray(onc_masses)
            onc_masses_error = 100. * numpy.asarray(onc_masses_error)

            if log:
                onc_masses = onc_masses[onc_masses > 0.0]
                sorted_onc_masses = numpy.sort(numpy.log10(onc_masses))
                sorted_onc_masses_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(onc_masses, onc_masses_error))])
            else:
                sorted_onc_masses = numpy.sort(onc_masses)
                sorted_onc_masses_errors = numpy.array([x for _, x in sorted(zip(onc_masses, onc_masses_error))])

            p = 1. * numpy.arange(len(sorted_onc_masses)) / (len(sorted_onc_masses) - 1)

            onc_low = sorted_onc_masses - sorted_onc_masses_errors
            onc_high = sorted_onc_masses + sorted_onc_masses_errors

            axs00.plot(sorted_onc_masses, p,
                           ls='-', lw=3,
                           color=colors[0],
                           label=labels[0])
            axs00.fill_betweenx(p,
                                onc_low, onc_high,
                                alpha='0.2', facecolor=colors[0])
            axs00.set_title('ONC')
            #axs00.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)

            # OMC-2 data (van Terwisga+ 2019)
            # Data: DUST masses in MEarth
            lines = open('data/OMC-2_masses.txt', 'r').readlines()
            omc2_masses, omc2_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split()
                a = data[0]
                b = data[2]

                # Unit conversion
                me = float(a) | units.MEarth
                mj = me.value_in(units.MJupiter)

                me_error = float(b) | units.MEarth
                mj_error = me_error.value_in(units.MJupiter)

                omc2_masses.append(mj)
                omc2_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            omc2_masses = 100. * numpy.asarray(omc2_masses)
            omc2_masses_error = 100. * numpy.asarray(omc2_masses_error)

            if log:
                omc2_masses = omc2_masses[omc2_masses > 0.0]
                sorted_omc2_masses = numpy.sort(numpy.log10(omc2_masses))
                sorted_omc2_masses_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(omc2_masses, omc2_masses_error))])
            else:
                sorted_omc2_masses = numpy.sort(omc2_masses)
                sorted_omc2_masses_errors = numpy.array([x for _, x in sorted(zip(omc2_masses, omc2_masses_error))])

            p = 1. * numpy.arange(len(sorted_omc2_masses)) / (len(sorted_omc2_masses) - 1)

            omc2_low = sorted_omc2_masses - sorted_omc2_masses_errors
            omc2_high = sorted_omc2_masses + sorted_omc2_masses_errors

            axs20.plot(sorted_omc2_masses, p,
                           ls='-', lw=3,
                           color=colors[6],
                           label=labels[6])
            axs20.fill_betweenx(p,
                                omc2_low, omc2_high,
                                alpha='0.2', facecolor=colors[6])
            axs20.set_title('OMC-2')
            #axs20.legend()
            axs20.set_xlabel(xlabel)
            axs20.set_ylabel(ylabel)

            # OMC-1 data (Eisner+ 2016)
            # Data: 100 * DUST masses in MJup
            lines = open('data/OMC-1_masses.txt', 'r').readlines()
            omc1_masses, omc1_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split()
                a = data[12]
                b = data[13]

                # No unit conversion needed
                omc1_masses.append(float(a))
                omc1_masses_error.append(float(b))

            omc1_masses = numpy.array(omc1_masses)
            omc1_masses_error = numpy.array(omc1_masses_error)

            if log:
                omc1_masses = omc1_masses[omc1_masses > 0.0]
                sorted_omc1_masses = numpy.sort(numpy.log10(omc1_masses))
                sorted_omc1_masses_errors = numpy.array([numpy.log10(x) for _, x in sorted(zip(omc1_masses, omc1_masses_error))])
            else:
                sorted_omc1_masses = numpy.sort(omc1_masses)
                sorted_omc1_masses_errors = numpy.array([x for _, x in sorted(zip(omc1_masses, omc1_masses_error))])

            p = 1. * numpy.arange(len(sorted_omc1_masses)) / (len(sorted_omc1_masses) - 1)

            omc1_low = sorted_omc1_masses - sorted_omc1_masses_errors
            omc1_high = sorted_omc1_masses + sorted_omc1_masses_errors

            axs21.plot(sorted_omc1_masses, p,
                           ls='-', lw=3,
                           color=colors[7],
                           label=labels[7])
            axs21.fill_betweenx(p,
                                omc1_low, omc1_high,
                                alpha='0.2', facecolor=colors[7])
            axs21.set_title('OMC-1')
            #axs21.legend()
            axs21.set_xlabel(xlabel)
            axs21.set_ylabel(ylabel)

        elif t == 2.0:
            axs01.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs01.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs01.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs01.set_xlim(xlimits)
            axs01.set_ylim(ylimits)
            axs01.set_xticks(ticks)

            # Lupus data (Ansdell+ 2018 2018ApJ...859...21A)
            # Data: GAS masses in MJup
            lupus_masses, lupus_low, lupus_high = [], [], []

            # No unit conversion needed
            lines = open('data/Lupus_masses.txt', 'r').readlines()

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[5]
                b = line.split()[6]
                c = line.split()[7]

                try:
                    lupus_masses.append(float(a))

                    if b == '...':
                        lupus_low.append(float(a))
                    else:
                        lupus_low.append(float(b))

                    if c == '...':
                        lupus_high.append(float(a))
                    else:
                        lupus_high.append(float(c))

                except ValueError:
                    lupus_masses.append(float(a[1:]))

                    if b == '...':
                        lupus_low.append(float(a[1:]))
                    else:
                        lupus_low.append(float(b))

                    if c == '...':
                        lupus_high.append(float(a[1:]))
                    else:
                        lupus_high.append(float(c))

            lupus_masses = numpy.array(lupus_masses)

            if log:
                lupus_masses = lupus_masses[lupus_masses > 0.0]
                lupus_sorted_masses = numpy.sort(numpy.log10(lupus_masses))
                lupus_sorted_low = numpy.array([numpy.log10(x) for _, x in sorted(zip(lupus_masses, lupus_low))])
                lupus_sorted_high = numpy.array([numpy.log10(x) for _, x in sorted(zip(lupus_masses, lupus_high))])
            else:
                lupus_sorted_masses = numpy.sort(lupus_masses)
                lupus_sorted_low = numpy.array([x for _, x in sorted(zip(lupus_masses, lupus_low))])
                lupus_sorted_high = numpy.array([x for _, x in sorted(zip(lupus_masses, lupus_high))])

            p = 1. * numpy.arange(len(lupus_sorted_masses)) / (len(lupus_sorted_masses) - 1)
            axs01.plot(lupus_sorted_masses, p,
                           ls='-', lw=3,
                           color=colors[1],
                           label=labels[1])
            axs01.fill_betweenx(p,
                                    lupus_sorted_low, lupus_sorted_high,
                                    alpha='0.2', facecolor=colors[1])
            axs01.set_title('Lupus clouds')
            #axs01.legend()
            axs01.set_xlabel(xlabel)
            axs01.set_ylabel(ylabel)

        elif t == 2.5:
            axs02.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs02.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs02.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs02.set_xlim(xlimits)
            axs02.set_ylim(ylimits)
            axs02.set_xticks(ticks)

            # Chamaeleon I data (Mulders et al 2017 2017ApJ...847...31M)
            # Data: DUST masses in LOG(MEarth)
            lines = open('data/ChamI_masses.txt', 'r').readlines()
            cham_masses, cham_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):  # DUST masses
                a = line.split()[8]
                b = line.split()[9]

                # MEarth to MJup conversion
                me = numpy.power(10, float(a)) | units.MEarth  # Data is in log
                mj = me.value_in(units.MJupiter)

                me_error = numpy.power(10, float(b)) | units.MEarth
                mj_error = me_error.value_in(units.MJupiter)

                # Saving data 10^
                cham_masses.append(mj)
                cham_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            cham_masses = 100. * numpy.asarray(cham_masses)
            cham_masses_error = 100. * numpy.asarray(cham_masses_error)

            if log:
                cham_masses = cham_masses[cham_masses > 0]
                cham_sorted_masses = numpy.sort(numpy.log10(cham_masses))
                cham_sorted_error = numpy.array([numpy.log10(x) for _, x in sorted(zip(cham_masses, cham_masses_error))])
            else:
                cham_sorted_masses = numpy.sort(cham_masses)
                cham_sorted_error = numpy.array([x for _, x in sorted(zip(cham_masses, cham_masses_error))])

            p = 1. * numpy.arange(len(cham_sorted_masses)) / (len(cham_sorted_masses) - 1)
            cham_low = cham_sorted_masses - cham_sorted_error
            cham_high = cham_sorted_masses + cham_sorted_error

            axs02.plot(cham_sorted_masses, p,
                           ls='-', lw=3,
                           color=colors[2],
                           label=labels[2])
            axs02.fill_betweenx(p,
                                cham_low, cham_high,
                                alpha='0.2', facecolor=colors[2])
            axs02.set_title('Chamaeleon I')
            #axs02.legend()
            axs02.set_xlabel(xlabel)
            axs02.set_ylabel(ylabel)

            # IC 348
            axs12.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs12.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs12.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs12.set_xlim(xlimits)
            axs12.set_ylim(ylimits)
            axs12.set_xticks(ticks)

            # IC 348 data (Ruiz-Rodriguez et al 2018  2018MNRAS.478.3674R )
            # Data: DUST masses in MEarth
            lines = open('data/IC348_masses.txt', 'r').readlines()
            ic348_masses, ic348_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                if len(line.split()) == 11:
                    a = line.split()[8]
                    b = line.split()[10]
                else:
                    a = line.split()[10]
                    b = line.split()[12]

                # MEarth to MJup conversion
                me = float(a) | units.MEarth
                mj = me.value_in(units.MJupiter)

                me_error = float(b) | units.MEarth
                mj_error = me_error.value_in(units.MJupiter)

                ic348_masses.append(mj)
                ic348_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            ic348_masses = 100. * numpy.asarray(ic348_masses)
            ic348_masses_error = 100. * numpy.asarray(ic348_masses_error)

            if log:
                ic348_masses = ic348_masses[ic348_masses > 0.0]
                ic348_sorted_masses = numpy.sort(numpy.log10(ic348_masses))
                ic348_sorted_error = numpy.array([numpy.log10(x) for _, x in sorted(zip(ic348_masses, ic348_masses_error))])
            else:
                ic348_sorted_masses = numpy.sort(ic348_masses)
                ic348_sorted_error = numpy.array([x for _, x in sorted(zip(ic348_masses, ic348_masses_error))])

            p = 1. * numpy.arange(len(ic348_sorted_masses)) / (len(ic348_sorted_masses) - 1)
            ic348_low = ic348_sorted_masses - ic348_sorted_error
            ic348_high = ic348_sorted_masses + ic348_sorted_error

            axs12.plot(ic348_sorted_masses, p,
                        ls='-', lw=3,
                        color=colors[5],
                        label=labels[5])
            axs12.fill_betweenx(p,
                                ic348_low, ic348_high,
                                alpha='0.2', facecolor=colors[5])
            axs12.set_title('IC 348')
            #axs02.legend()
            axs12.set_xlabel(xlabel)
            axs12.set_ylabel(ylabel)

        elif t == 4.0:
            axs10.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs10.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs10.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs10.set_xlim(xlimits)
            axs10.set_ylim(ylimits)
            axs10.set_xticks(ticks)

            # sigma Orionis data (Ansdell+ 2017  2017AJ....153..240A)
            # Data: DUST masses in MEarth
            lines = open('data/sigmaOrionis_masses.txt', 'r').readlines()
            sOrionis_masses, sOrionis_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[15]
                b = line.split()[16]

                # MEarth to MJup conversion
                me = float(a) | units.MEarth
                mj = me.value_in(units.MJupiter)

                me_error = float(b) | units.MEarth
                mj_error = me_error.value_in(units.MJupiter)

                sOrionis_masses.append(mj)
                sOrionis_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            sOrionis_masses = 100. * numpy.asarray(sOrionis_masses)
            sOrionis_masses_error = 100. * numpy.asarray(sOrionis_masses_error)

            if log:
                sOrionis_masses = sOrionis_masses[sOrionis_masses > 0.0]
                sOrionis_sorted_masses = numpy.sort(numpy.array(numpy.log10(sOrionis_masses)))
                sOrionis_sorted_error = numpy.array([numpy.log10(x) for _, x in sorted(zip(sOrionis_masses, sOrionis_masses_error))])
            else:
                sOrionis_sorted_masses = numpy.sort(numpy.array(sOrionis_masses))
                sOrionis_sorted_error = numpy.array([x for _, x in sorted(zip(sOrionis_masses, sOrionis_masses_error))])

            p = 1. * numpy.arange(len(sOrionis_sorted_masses)) / (len(sOrionis_sorted_masses) - 1)
            sOrionis_low = sOrionis_sorted_masses - sOrionis_sorted_error
            sOrionis_high = sOrionis_sorted_masses + sOrionis_sorted_error

            axs10.plot(sOrionis_sorted_masses, p,
                           ls='-', lw=3,
                           color=colors[3],
                           label=labels[3])
            axs10.fill_betweenx(p,
                                    sOrionis_low, sOrionis_high,
                                    alpha='0.2', facecolor=colors[3])
            axs10.set_title('$\sigma$ Orionis')
            #axs10.legend()
            axs10.set_xlabel(xlabel)
            axs10.set_ylabel(ylabel)

        elif t == 5.0:
            axs11.plot(disk_masses,
                        cumulative_masses,
                        lw=2, color='black')
            axs11.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs11.text(xtext, ytext, 't = {0} Myr'.format(t))
            axs11.set_xlim(xlimits)
            axs11.set_ylim(ylimits)
            axs11.set_xticks(ticks)

            # UpperSco data (Barenfeld et al 2016 2016ApJ...827..142B)
            # data: DUST masses in MEarth
            lines = open('data/UpperSco_masses.txt', 'r').readlines()
            uppersco_masses, uppersco_masses_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[1]
                b = line.split()[2]

                try:
                    me = float(a) | units.MEarth  # MEarth to MJup conversion
                    mj = me.value_in(units.MJupiter)
                    uppersco_masses.append(mj)
                except ValueError:
                    me = float(a[1:]) | units.MEarth  # MEarth to MJup conversion
                    mj = me.value_in(units.MJupiter)
                    uppersco_masses.append(mj)

                if b == "...":
                    uppersco_masses_error.append(0.0)
                else:
                    me_error = float(b) | units.MEarth
                    mj_error = me_error.value_in(units.MJupiter)
                    uppersco_masses_error.append(mj_error)

            # 100. factor to turn dust mass into gas mass
            uppersco_masses = 100. * numpy.asarray(uppersco_masses)
            uppersco_masses_error = 100. * numpy.asarray(uppersco_masses_error)

            if log:
                uppersco_masses = uppersco_masses[uppersco_masses > 0.0]
                uppersco_sorted_masses = numpy.sort(numpy.log10(uppersco_masses))
                uppersco_sorted_error = numpy.array([numpy.log10(x) for _, x in sorted(zip(uppersco_masses, uppersco_masses_error))])
            else:
                uppersco_sorted_masses = numpy.sort(uppersco_masses)
                uppersco_sorted_error = numpy.array([x for _, x in sorted(zip(uppersco_masses, uppersco_masses_error))])

            p = 1. * numpy.arange(len(uppersco_sorted_masses)) / (len(uppersco_sorted_masses) - 1)
            uppersco_low = uppersco_sorted_masses - uppersco_sorted_error
            uppersco_high = uppersco_sorted_masses + uppersco_sorted_error

            axs11.plot(uppersco_sorted_masses, p,
                           ls='-', lw=3,
                           color=colors[4],
                           label=labels[4])
            axs11.fill_betweenx(p,
                                    uppersco_low, uppersco_high,
                                    alpha='0.2', facecolor=colors[4])
            axs11.set_title('UpperSco')
            #axs11.legend()
            axs11.set_xlabel(xlabel)
            axs11.set_ylabel(ylabel)

    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=1.5, hspace=2.5)  # make the figure look better
    pyplot.savefig('{0}/CDF_data_mass.png'.format(save_path))

    pyplot.show()


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


def disk_fractions(N, open_paths, save_path):
    filename = 'data/diskfractions.dat'
    f = open(filename, "r")
    lines = f.readlines()
    ages, ages_errors, disk_fraction, df_lower, df_higher = [], [], [], [], []
    src1_count = 0

    label1 = "Ribas et al 2014"
    label2 = "Richert et al 2018"

    print matplotlib.matplotlib_fname()

    for l in lines:
        li = l.strip()
        if not li.startswith("#"):
            x = l.split()
            ages.append(float(x[1]))
            ages_errors.append(float(x[2]))

            if int(x[6]) == 1:
                src1_count += 1
                disk_fraction.append(float(x[3]))
                df_lower.append(float(x[4]))
                df_higher.append(float(x[5]))
            else:
                disk_fraction.append(float(x[3]) * 100)
                df_lower.append(float(x[4]) * 100)
                df_higher.append(float(x[5]) * 100)

    f.close()

    # Separating by paper source
    ages1 = ages[:src1_count]
    ages2 = ages[src1_count:]
    ages_errors1 = ages_errors[:src1_count]
    ages_errors2 = ages_errors[src1_count:]
    disk_fraction1 = disk_fraction[:src1_count]
    disk_fraction2 = disk_fraction[src1_count:]
    df_lower1 = df_lower[:src1_count]
    df_lower2 = df_lower[src1_count:]
    df_higher1 = df_higher[:src1_count]
    df_higher2 = df_higher[src1_count:]

    df_errors1 = numpy.array((df_lower1, df_higher1))
    df_errors2 = numpy.array((df_lower2, df_higher2))

    fig = pyplot.figure(figsize=(12, 12))
    markers1, caps1, bars1 = pyplot.errorbar(ages1,
                                             disk_fraction1,
                                             xerr=ages_errors1,
                                             yerr=df_errors1,
                                             fmt='o', lw=1, alpha=0.5,
                                             label=label1)
    markers2, caps2, bars2 = pyplot.errorbar(ages2,
                                             disk_fraction2,
                                             xerr=ages_errors2,
                                             yerr=df_errors2,
                                             fmt='o', lw=1, alpha=0.5,
                                             label=label2)

    [bar.set_alpha(0.5) for bar in bars1]
    [bar.set_alpha(0.5) for bar in bars2]

    # Plotting my data
    times = numpy.arange(0.0, 5.5, 0.5)
    all_fractions = []

    for p in open_paths:
        fractions = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.stellar_mass.value_in(units.MSun) <= 1.9]
            disked_stars = small_stars[small_stars.dispersed == False]

            #density = 1E-7

            #small_stars = small_stars[small_stars.disk_mass.value_in(units.MSun) /
            #                          (numpy.pi * small_stars.disk_radius.value_in(units.au) ** 2) > density]
            #disked_stars2 = small_stars[small_stars.dispersed == False]
            #disked_stars = disked_stars2[disked_stars2.photoevap_mass_loss < disked_stars2.initial_disk_mass]

            #print len(disked_stars2), len(disked_stars)

            fraction = 100. * (float(len(disked_stars)) / float(len(small_stars)))
            print fraction
            fractions.append(fraction)
        all_fractions.append(fractions)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_stdev = numpy.std(all_fractions, axis=0)

    pyplot.errorbar(times,
                    all_disk_fractions,
                    yerr=disk_fractions_stdev,
                    marker='o', color='k')


    pyplot.legend()
    pyplot.xlabel("Age [Myr]")
    pyplot.ylabel("Disk fraction [\%]")
    pyplot.savefig('{0}/disk_fraction.png'.format(save_path))

    pyplot.show()


def main(run_number, save_path, time, N, distribution, ncells, density, i, all_distances, single):

    files = ['results/king_N100_c100_3/0/N100_t{0}.hdf5'.format(time)]#,
             #'results/plummer_N100_c100_3/0/N100_t10.0.hdf5']#,
             #'results/fractal_N100_c50/0/N100_t0.hdf5']

    paths = ['results/plummer_N100_c50_sh/0/',
             'results/cartesius/plummer_N100_c50/0/']#,
        #     'results/plummer_N{0}_c{1}_3/0/'.format(N, ncells)]#,
             #'results/fractal_N{0}_c{1}_3/0/']

    labels = ["King"]#, "Plummer"]#, "Fractal"]
    plot_colors = ["#ca5670", "#638ccc", "#c57c3c", "#72a555", "#ab62c0"]

    #size_vs_mass(files, labels, plot_colors, density, N, ncells, 10.0)
    #size_vs_distance_from_star(paths, time, N, labels, plot_colors, density)
    #distance_from_center(paths, time, N, labels, plot_colors, density)

    #path = 'results/cartesius/{0}_N{1}_c{2}/0/'.format(distribution, N, ncells)
    path = 'results/cartesius/plummer_N100_c50/0/'.format(distribution, N, ncells)

    pyplot.style.use('paper')

    if single:
        single_star(path, save_path, N, i, time, all_distances)
    #pyplot.show()
    else:
        #disk_fractions(N, paths, save_path)
        times = [1.0, 2.0, 2.5, 4.0, 5.0]
        #colors = ['#E24A33', '#348ABD', '#988ED5', '#8EBA42', '#FFB5B8', '#FBC15E', '#777777']
        #colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442'] #seaborn color blind
        colors = ["#638ccc", "#ca5670", "#c57c3c", "#72a555", "#ab62c0", '#0072B2', '#009E73', '#D55E00']  # colors from my prev paper
        labels = ['Trapezium cluster', 'Lupus clouds', 'Chamaeleon I', '$\sigma$ Orionis', 'Upper Scorpio', 'IC 348',
                  'ONC', "OMC-2"]
        #cdfs_in_time(path, save_path, N, times)
        #cdfs_with_observations_size(paths, save_path, N, times, colors, labels)
        #cdfs_with_observations_mass(paths, save_path, N, times, colors, labels, log=True)

        # For presentation
        #cdfs_w_old(paths, labels, plot_colors, density, N, time)
        #size_vs_mass(files, labels, plot_colors, density, N, ncells, time)

        #cdfs(paths, save_path, N, time)
        #size_vs_distance_from_star(paths, time, N, labels, plot_colors, density)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation parameters
    result.add_option("-n", dest="run_number", type="int", default=0,
                      help="run number [%default]")
    result.add_option("-s", dest="save_path", type="string", default='/media/fran/data1/photoevap/figures',
                      help="path to save the results [%default]")
    result.add_option("-t", dest="time", type="float", default='10.0',
                      help="time to use for plots [%default]")

    # Cluster parameters
    result.add_option("-N", dest="N", type="int", default=100,
                      help="number of stars [%default]")
    result.add_option("-p", dest="distribution", type="string", default='plummer',
                      help="spatial distribution to plot [%default]")
    result.add_option("-a", dest="all_distances", type="int", default=0,
                      help="distances to bright stars [%default]")
    result.add_option("-b", dest="single", type="int", default=0,
                      help="single star plots [%default]")

    # Disk parameters
    result.add_option("-c", dest="ncells", type="int", default=100,
                      help="Number of cells to be used in vader disk [%default]")
    result.add_option("-d", dest="density", type="float", default=1E-9,
                      help="Density limit for disk surface [%default]")
    result.add_option("-i", dest="i", type="int", default=0,
                      help="Individual star to plot [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

