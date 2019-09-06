import numpy
import math
import matplotlib
from matplotlib import pyplot
from amuse.lab import *
from amuse import io
import matplotlib.lines as mlines
import matplotlib.patches as patches


# Custom legends
class PhotoevapObject(object):
    pass


class TruncationObject(object):
    pass


class M100Object(object):
    pass


class M30Object(object):
    pass


class M100shadedObject(object):
    pass


class M30shadedObject(object):
    pass


class PhotoevapObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color="#009bed")
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color="#009bed")
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class TruncationObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color="#d73027")
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color="#d73027")
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class M100ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color="black")
        handlebox.add_artist(l1)
        return [l1]


class M30ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color="black")
        handlebox.add_artist(l1)
        return [l1]


class M100shadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color='black')  # Have to change color by hand for other plots
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 43),  # (x,y)
            1.2 * width,  # width
            1.4 * height,  # height
            fill='black',
            facecolor='black',
            #edgecolor="black",
            alpha=0.2,
            #hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class M30shadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color='black')
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 43),  # (x,y)
            1.15 * width,  # width
            1.4 * height,  # height
            fill='black',
            facecolor='black',
            edgecolor='black',
            alpha=0.2,
            hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


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


def get_disk_radius(disk, density_limit=1E-10):
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


def radius_plot():
    """ Figure 1

    :return:
    """
    disk = initialize_vader_code(100 | units.au, 0.1 | units.MSun, 1E-5, r_max=rout, n_cells=100, linear=False)

    #print disk_radii
    pyplot.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), color='k', lw=2, label='t = 0.0 Myr')
    pyplot.axvline(get_disk_radius(disk, density_limit=1E-8).value_in(units.au), color='red', lw=2)

    disk.evolve_model(0.1 | units.Myr)

    pyplot.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), color='k', lw=2, ls=':', label='t = 0.1 Myr')
    pyplot.axvline(get_disk_radius(disk, density_limit=1E-11).value_in(units.au), color='red', lw=2, ls=':')

    pyplot.xlim([0, 1000])
    pyplot.xlabel('Disk radius [au]')
    pyplot.ylabel('Surface density [g / cm$^2$]')
    pyplot.legend(loc='lower left')
    pyplot.savefig('radii_density.png')
    pyplot.show()


def luminosity_fit(masses):
    """
    Return stellar luminosity (in LSun) for corresponding mass, as calculated with Martijn's fit

    :param mass: stellar mass in MSun
    :return: stellar luminosity in LSun
    """
    fit = []

    for mass in masses:
        if 0.12 <= mass < 0.24:
            fit.append(1.70294E16 * numpy.power(mass, 42.557)) #| units.LSun
        elif 0.24 <= mass < 0.56:
            fit.append(9.11137E-9 * numpy.power(mass, 3.8845)) #| units.LSun
        elif 0.56 <= mass < 0.70:
            fit.append(1.10021E-6 * numpy.power(mass, 12.237)) #| units.LSun
        elif 0.70 <= mass < 0.91:
            fit.append(2.38690E-4 * numpy.power(mass, 27.199)) #| units.LSun
        elif 0.91 <= mass < 1.37:
            fit.append(1.02477E-4 * numpy.power(mass, 18.465)) #| units.LSun
        elif 1.37 <= mass < 2.07:
            fit.append(9.66362E-4 * numpy.power(mass, 11.410)) #| units.LSun
        elif 2.07 <= mass < 3.72:
            fit.append(6.49335E-2 * numpy.power(mass, 5.6147)) #| units.LSun
        elif 3.72 <= mass < 10.0:
            fit.append(6.99075E-1 * numpy.power(mass, 3.8058)) #| units.LSun
        elif 10.0 <= mass < 20.2:
            fit.append(9.73664E0 * numpy.power(mass, 2.6620)) #| units.LSun
        elif 20.2 <= mass:
            fit.append(1.31175E2 * numpy.power(mass, 1.7974)) #| units.LSun
        else:
            fit.append(0.) #| units.LSun

    return fit


def parravano_fit(masses):
    """ Parravano 2003 luminosity fit for mass-FUV luminosity plot

    :param mass:
    :return:
    """
    fit = []

    for mass in masses:
        if 0.12 <= mass < 2.:
            fit.append(2.77 * 1E-4 * numpy.power(mass, 11.8)) #| units.LSun
        elif 2. <= mass < 2.5:
            fit.append(1.88 * 1E-3 * numpy.power(mass, 9.03)) #| units.LSun
        elif 2.5 <= mass < 3.:
            fit.append(1.19 * 1E-2 * numpy.power(mass, 7.03)) #| units.LSun
        elif 3. <= mass < 6.:
            fit.append(1.47 * 1E-1 * numpy.power(mass, 4.76)) #| units.LSun
        elif 6. <= mass < 9.:
            fit.append(8.22 * 1E-1 * numpy.power(mass, 3.78)) #| units.LSun
        elif 9. <= mass < 12.:
            fit.append(2.29 * 1E0 * numpy.power(mass, 3.31)) #| units.LSun
        elif 12. <= mass < 30.:
            fit.append(2.70 * 1E1 * numpy.power(mass, 2.32)) #| units.LSun
        elif 30. <= mass < 120.:
            fit.append(3.99 * 1E2 * numpy.power(mass, 1.54)) #| units.LSun
        else:
            fit.append(0) #| units.LSun

    return fit


def luminosity_vs_mass(save_path, save):
    """ Figure 2

    :param save_path:
    :param save:
    :return:
    """
    masses = numpy.arange(0.12, 100., 0.2)

    this_fit = luminosity_fit(masses)
    parravano = parravano_fit(masses)

    fig = pyplot.figure()
    ax = pyplot.gca()

    pyplot.loglog(masses, parravano, lw=3, label="Parravano et al (2003)", color='#fe6950')
    pyplot.loglog(masses, this_fit, lw=3, label="This work", color='#002c94')
    ax.axvline(1.9, color='black', ls=':')
    pyplot.text(1.55, 1E-15, 'M$_*$ = 1.9 M$_\odot$', rotation=90)
    pyplot.legend(loc='lower right')
    pyplot.xlabel('Stellar mass [M$_\odot$]')
    pyplot.ylabel('FUV luminosity [L$_\odot$]')
    pyplot.xlim([0, 1E2])
    #pyplot.ylim([0, 1E6])
    pyplot.xticks([1, 10, 100])
    if save:
        pyplot.savefig('{0}/luminosity_fit.png'.format(save_path))
    pyplot.show()


def mass_loss_in_time(open_paths100, open_paths50, save_path, tend, save, mass_limit=0.0):
    """ Cumulative mass loss in each time step due to photoevaporation and truncations.

    :param open_path:
    :param save_path:
    :param tend:
    :param N:
    :param i:
    :return:
    """
    times = numpy.arange(0.0, tend + 0.05, 0.05)

    # Small fix because I kept saving mas losses after disks were dispersed
    """for t in times:
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.bright == False]
            for s in small_stars:
                if s.dispersed:
                    s.photoevap_mass_loss = 0.0 | units.MJupiter
                    s.truncation_mass_loss = 0.0 | units.MJupiter
            io.write_set_to_file(stars, f, 'hdf5')

    for t in times:
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.bright == False]
            for s in small_stars:
                if s.dispersed:
                    s.photoevap_mass_loss = 0.0 | units.MJupiter
                    s.truncation_mass_loss = 0.0 | units.MJupiter
            io.write_set_to_file(stars, f, 'hdf5')"""

    fig = pyplot.figure()
    ax = pyplot.gca()
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'Mean mass loss [M$_{Jup}$]')

    photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []
    disk_mass, mass_low, mass_high = [], [], []

    for t in times:
        photoevap_in_t, trunc_in_t = [], []
        mass_in_t = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            small_stars = stars[stars.bright == False]
            #small_stars = small_stars1[small_stars1.dispersed == False]

            photoevap_in_t.append(numpy.mean(small_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)))
            trunc_in_t.append(numpy.mean(small_stars.cumulative_truncation_mass_loss.value_in(units.MJupiter)))
            mass_in_t.append(numpy.mean(small_stars.disk_mass.value_in(units.MJupiter)))

            #if t == 0.05:
            #    for s in small_stars1:
            #        print s.initial_disk_mass.in_(units.MSun), s.dispersed
            #    print "*"

        photoevap_mass_loss.append(numpy.mean(photoevap_in_t))
        photoevap_low.append(numpy.min(photoevap_in_t))
        photoevap_high.append(numpy.max(photoevap_in_t))

        trunc_mass_loss.append(numpy.mean(trunc_in_t))
        trunc_low.append(numpy.min(trunc_in_t))
        trunc_high.append(numpy.max(trunc_in_t))

        disk_mass.append(numpy.mean(mass_in_t))
        mass_low.append(numpy.min(mass_in_t))
        mass_high.append(numpy.max(mass_in_t))

    ax.plot(times, photoevap_mass_loss, label="Photoevaporation", lw=3, color="#009bed")
    ax.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    facecolor="#009bed",
                    alpha=0.2)

    ax.plot(times, trunc_mass_loss, label="Dynamical truncations", lw=3, color="#d73027")
    ax.fill_between(times,
                    trunc_low,
                    trunc_high,
                    facecolor="#d73027",
                    alpha=0.2)

    """ax.semilogx(times, disk_mass, label="Mean mass", lw=3, color="black")
    ax.fill_between(times,
                    mass_low,
                    mass_high,
                    facecolor="black",
                    alpha=0.2)"""

    photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []
    disk_mass, mass_low, mass_high = [], [], []

    for t in times:
        photoevap_in_t, trunc_in_t = [], []
        mass_in_t = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            small_stars = stars[stars.bright == False]
            #small_stars = small_stars1[small_stars1.dispersed == False]

            photoevap_in_t.append(numpy.mean(small_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)))
            trunc_in_t.append(numpy.mean(small_stars.cumulative_truncation_mass_loss.value_in(units.MJupiter)))
            mass_in_t.append(numpy.mean(small_stars.disk_mass.value_in(units.MJupiter)))

        photoevap_mass_loss.append(numpy.mean(photoevap_in_t))
        photoevap_low.append(numpy.min(photoevap_in_t))
        photoevap_high.append(numpy.max(photoevap_in_t))

        trunc_mass_loss.append(numpy.mean(trunc_in_t))
        trunc_low.append(numpy.min(trunc_in_t))
        trunc_high.append(numpy.max(trunc_in_t))

        disk_mass.append(numpy.mean(mass_in_t))
        mass_low.append(numpy.min(mass_in_t))
        mass_high.append(numpy.max(mass_in_t))

    ax.plot(times, photoevap_mass_loss, label="Photoevaporation", ls="--", lw=3, color="#009bed")
    ax.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    alpha=0.2, facecolor="#009bed", edgecolor='#009bed', hatch="/")

    ax.plot(times, trunc_mass_loss, label="Dynamical truncations", ls="--", lw=3, color="#d73027")
    ax.fill_between(times,
                    trunc_low,
                    trunc_high,
                    alpha=0.2, facecolor="#d73027", edgecolor='#d73027', hatch="/")

    """ax.semilogx(times, disk_mass, label="Mean mass", ls="--", lw=3, color="black")
    ax.fill_between(times,
                    mass_low,
                    mass_high,
                    facecolor="black",
                    alpha=0.2)"""

    ax.set_xlim([0.0, 5.0])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    ax.legend([PhotoevapObject(), TruncationObject(), M100shadedObject, M30shadedObject],
               ['Photoevaporation', 'Dynamical truncations',
                r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
               handler_map={PhotoevapObject: PhotoevapObjectHandler(),
                            TruncationObject: TruncationObjectHandler(),
                            M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
               loc='best', bbox_to_anchor=(0.85, -0.15), ncol=2,
              fontsize=20, framealpha=1.)
    ax.tick_params(which='minor', direction='out', length=6, width=1)
    ax.tick_params(which='major', direction='out', length=6, width=1)
    #pyplot.xticks([0.1, 1, 5], ['0.1', '1', '5'])

    #pyplot.tight_layout()
    if save:
        pyplot.savefig('{0}/mass_loss.png'.format(save_path))
    pyplot.show()


def single_star(open_path, save_path, N, k, t_end, all_distances=0):
    #fig = pyplot.figure(figsize=(12, 8))
    #ax = pyplot.gca()
    fig, axs = pyplot.subplots(3, 1, figsize=(8, 16), sharex=True)

    ax = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    #ax4 = axs[1, 1]

    # Mass loss
    #ax.set_title('Mass loss in time')
    #ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'Mass loss [M$_{Jup}$]')
    #ax.set_xlim(0, t_end)

    # Distance
    #ax2.set_title('Distance to bright star(s)')
    #ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Distance [parsec]')

    # Disk size
    #ax3.set_title('Disk size')
    ax3.set_xlabel('Time [Myr]')
    ax3.set_ylabel('Disk size [au]')
    #ax3.set_xlim([0, t_end])

    # G0
    #ax4.set_title(r'$G_0$')
    #ax4.set_xlabel('Time [Myr]')
    #ax4.set_ylabel(r'$G_0$')

    times = numpy.arange(0.00, t_end + 0.05, 0.05)

    disk_sizes = []
    initial_size = 0.0

    mass_loss_pe, mass_loss_trunc = [], []

    dispersal_time = 0.
    checked = False

    brighter = []
    brighter_masses = []

    mass = 0.

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        bright_stars = stars[stars.bright == True]
        s = stars[k]
        if t == 0.:
            mass = s.stellar_mass.value_in(units.MSun)
        #prev_ml_pe = s.photoevap_mass_loss.value_in(units.MJupiter)
        #prev_ml_trunc = s.truncation_mass_loss.value_in(units.MJupiter)
        #cum_ml_pe += prev_ml_pe
        #cum_ml_trunc += prev_ml_trunc
        #print "MASS LOSS: {0} MJup".format(s.photoevap_mass_loss.value_in(units.MJupiter) + s.truncation_mass_loss.value_in(units.MJupiter))
        #print "CUMULATIVE MASS LOSS: {0} MJup".format(s.cumulative_photoevap_mass_loss.value_in(units.MJupiter))
        #print "INITIAL DISK MASS {0} MJup".format(s.initial_disk_mass.value_in(units.MJupiter))
        #print s.dispersed
        #print checked
        #print "Density: {0} g/cm-2".format(s.disk_mass.value_in(units.g) / numpy.pi * s.disk_radius.value_in(units.cm)**2)

        #print 't={0}, r={1}, m={2}, disp={3}, tdisp={4}'.format(t,
        #                                             s.disk_radius.value_in(units.au),
        #                                             s.disk_mass.value_in(units.MJupiter),
        #                                             s.dispersed,
        #                                             s.dispersal_time.value_in(units.Myr))

        if t == 0.0:
            initial_size = s.disk_radius.value_in(units.au)

            for b in range(len(bright_stars)):
                brighter.append([])
                brighter_masses.append([])

        if not s.dispersed:
            disk_sizes.append(s.disk_radius.value_in(units.au))
        else:
            if not checked:
                dispersal_time = t
                checked = True
            disk_sizes.append(0.)

        #fig.suptitle(r"$M_*$ = {0:.2f} $M_\odot$, $R_d$ = {1:.2f} au, $M_d$ = {2:.2f} $M_J$".format(mass, initial_size, s.initial_disk_mass.value_in(units.MJupiter)))

        mass_loss_pe.append(s.cumulative_photoevap_mass_loss.value_in(units.MJupiter))
        mass_loss_trunc.append(s.cumulative_truncation_mass_loss.value_in(units.MJupiter))


        for i in range(len(bright_stars)):
            bs = bright_stars[i]
            brighter[i].append(distance(s, bs).value_in(units.parsec))
            brighter_masses[i].append(bs.mass.value_in(units.MSun))

    ax3.plot(times, disk_sizes, lw=3, c='black')
    #ax3.axhline(initial_size, ls=":")
    #ax3.text(1.0, 7*initial_size/8, "Init radius = {0} au".format(initial_size))
    ax3.axvline(dispersal_time, ls=":", c='black', lw=2)

    ax.plot(times, mass_loss_pe, color="#009bed", lw=3, label="Photoevaporation")
    ax.plot(times, mass_loss_trunc, color="#d73027", lw=3, label="Dynamical truncations")
    ax.legend(loc='upper left', fontsize=14)
    ax.axvline(dispersal_time, ls=":", c='black', lw=2)

    colors = ["#72a555", "#D55E00", '#988ED5']

    #["#638ccc", "#ca5670", '#009E73', '#0072B2',
    #          "#ca5670", "#c57c3c", "#72a555", "#ab62c0", '#0072B2', '#009E73',
    #          '#D55E00']


    for j in range(len(bright_stars)):
        ax2.plot(times, brighter[j],
                 label="{0:.2} M$_\odot$".format(bright_stars[j].stellar_mass.value_in(units.MSun)),
                 lw=3, c=colors[j])
    ax2.axvline(dispersal_time, ls=":", c='black', lw=2)
    ax2.legend(loc='upper right', fontsize=14)

    pyplot.tight_layout()
    pyplot.xlim([0, 5])
    pyplot.subplots_adjust(hspace=0.)  # make the figure look better
    pyplot.savefig('{0}/{1}.png'.format(save_path, k))
    pyplot.close()
    #pyplot.show()


def disk_mass(open_paths100, open_paths50, save_path, t_end, save):
    """ Figure 7

    :param open_paths100:
    :param open_paths50:
    :param save_path:
    :param t_end:
    :param save:
    :return:
    """
    fig = pyplot.figure()

    total_disks, total_disks_low, total_disks_high = [], [], []
    times = numpy.arange(0.0, t_end + 0.05, 0.05)
    init_mass = 0.

    # 100 MSun
    for t in times:
        total_in_t, total_in_t100 = [], []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_masses = 1E-2 * small_stars.disk_mass.value_in(units.MEarth)

            masses = disk_masses[disk_masses > 10.]

            if t == 0.:
                init_mass = float(len(masses))

            total_in_t.append(len(masses) / init_mass)

        total_disks.append(numpy.mean(total_in_t))
        total_disks_low.append(numpy.min(total_in_t))
        total_disks_high.append(numpy.max(total_in_t))

    pyplot.plot(times,
                total_disks,
                lw=3,
                color='darkolivegreen',
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')#, capsize=5, facecolor='lightgray')
    pyplot.fill_between(times,
                        total_disks_low,
                        total_disks_high,
                        alpha=0.2, facecolor='darkolivegreen')

    # 50 MSun
    total_disks, total_disks_low, total_disks_high = [], [], []
    init_mass = 0.

    for t in times:
        total_in_t = []
        total_in_t100 = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_masses = small_stars.disk_mass.value_in(units.MEarth)

            masses = 1E-2 * disk_masses[disk_masses > 10.]
            #print t, len(masses)

            if t == 0.:
                init_mass = float(len(masses))

            total_in_t.append(len(masses) / init_mass)

        total_disks.append(numpy.mean(total_in_t))
        total_disks_low.append(numpy.min(total_in_t))
        total_disks_high.append(numpy.max(total_in_t))

    pyplot.plot(times,
                total_disks,
                lw=3, ls='--', color='darkolivegreen',
                label=r'$\rho \sim 50 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')#, capsize=5, facecolor='lightgray')
    pyplot.fill_between(times,
                        total_disks_low,
                        total_disks_high,
                        alpha=0.2, facecolor='darkolivegreen',
                        edgecolor='darkolivegreen', hatch="/")

    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel(r'$f_{> 10 \mathrm{\ M}_{\oplus}}$', fontsize=30)
    pyplot.legend([M100shadedObject, M30shadedObject],
                  [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                   r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
                handler_map={M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
                loc='best',# bbox_to_anchor=(0.73, -0.15), ncol=2,
                fontsize=22, framealpha=1.)
    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0.0, 1.0])
    if save:
        pyplot.savefig('{0}/mass_fraction_line.png'.format(save_path))
    pyplot.show()


def disk_size(open_paths100, open_paths50, save_path, t_end, save):
    """ Figure 8

    :param open_paths100:
    :param open_paths50:
    :param save_path:
    :param t_end:
    :param save:
    :return:
    """
    fig = pyplot.figure()

    total_disks, total_disks_low, total_disks_high = [], [], []
    times = numpy.arange(0.0, t_end + 0.05, 0.05)
    init_size = 0.

    for t in times:
        total_in_t = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_sizes = small_stars.disk_radius.value_in(units.au)

            sizes = disk_sizes[disk_sizes > 50.]

            if t == 0.:
                init_size = float(len(small_stars))

            total_in_t.append(len(sizes) / init_size)

        total_disks.append(numpy.mean(total_in_t))
        total_disks_low.append(numpy.min(total_in_t))
        total_disks_high.append(numpy.max(total_in_t))

    pyplot.plot(times,
                total_disks,
                lw=3,
                color='rebeccapurple',
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')#, capsize=5, facecolor='lightgray')
    pyplot.fill_between(times,
                        total_disks_low,
                        total_disks_high,
                        alpha=0.2, facecolor='rebeccapurple')

    total_disks, total_disks_low, total_disks_high = [], [], []
    init_size = 0.
    for t in times:
        total_in_t = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_sizes = small_stars.disk_radius.value_in(units.au)

            sizes = disk_sizes[disk_sizes > 50.]
            print sizes

            if t == 0.:
                init_size = float(len(small_stars))

            total_in_t.append(len(sizes) / init_size)

        total_disks.append(numpy.mean(total_in_t))
        total_disks_low.append(numpy.min(total_in_t))
        total_disks_high.append(numpy.max(total_in_t))

    pyplot.plot(times,
                total_disks,
                lw=3, ls='--', color='rebeccapurple',
                label=r'$\rho \sim 50 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')#, capsize=5, facecolor='lightgray')
    pyplot.fill_between(times,
                        total_disks_low,
                        total_disks_high,
                        alpha=0.2, facecolor='rebeccapurple',
                        edgecolor='rebeccapurple', hatch="/")

    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel(r'$f_{> 50 \mathrm{\ au}}$', fontsize=30)

    pyplot.legend([M100shadedObject, M30shadedObject],
                  [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                   r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
                handler_map={M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
                loc='best',# bbox_to_anchor=(0.73, -0.15), ncol=2,
                fontsize=22, framealpha=1.)

    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0.0, 1.0])
    if save:
        pyplot.savefig('{0}/size_fraction_line.png'.format(save_path))
    pyplot.show()


def count_stars(stars, d):
    n = 0
    for s in stars:
        if numpy.sqrt(s.x.value_in(units.parsec)**2
                      + s.y.value_in(units.parsec)**2
                      + s.z.value_in(units.parsec)**2) < d:
            n +=1
    return n


def disk_fractions(open_paths100, open_paths50, t_end, save_path, save, mass_limit=0.0):
    """ Figure 6 from paper.

    :param open_paths100: files for 100 MSun region
    :param open_paths50: files for 30 MSun region
    :param t_end: end time of simulations
    :param save_path: path to save figure
    :param save:
    :param mass_limit:
    :return:
    """
    # Plotting observations
    filename = 'data/diskfractions.dat'
    f = open(filename, "r")
    lines = f.readlines()
    ages, ages_errors, disk_fraction, df_lower, df_higher = [], [], [], [], []
    relax_times = []
    src1_count = 0
    Nobs = []

    label1 = "Ribas et al (2014)"
    label2 = "Richert et al (2018)"

    fig = pyplot.figure(figsize=(10, 10))

    for l in lines:
        li = l.strip()
        if not li.startswith("#"):
            x = l.split()
            ages.append(float(x[1]))
            ages_errors.append(float(x[2]))
            N = float(x[7])
            Nobs.append(N)
            relax_times.append(N / (6 * numpy.log(N)))  # "Instantaneous" relaxation time

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
    ages1 = numpy.array(ages[:src1_count])
    ages2 = numpy.array(ages[src1_count:])
    relax1 = numpy.array(relax_times[:src1_count])
    relax2 = numpy.array(relax_times[src1_count:])
    Nobs1 = numpy.array(Nobs[:src1_count])
    Nobs2 = numpy.array(Nobs[src1_count:])
    ages_errors1 = numpy.array(ages_errors[:src1_count])
    ages_errors2 = numpy.array(ages_errors[src1_count:])
    disk_fraction1 = numpy.array(disk_fraction[:src1_count])
    disk_fraction2 = numpy.array(disk_fraction[src1_count:])
    df_lower1 = numpy.array(df_lower[:src1_count])
    df_lower2 = numpy.array(df_lower[src1_count:])
    df_higher1 = numpy.array(df_higher[:src1_count])
    df_higher2 = numpy.array(df_higher[src1_count:])

    df_errors1 = numpy.array((df_lower1, df_higher1))
    df_errors2 = numpy.array((df_lower2, df_higher2))

    markers1, caps1, bars1 = pyplot.errorbar(ages1 / relax1,
                                             disk_fraction1 / 100.,
                                             xerr=ages_errors1 / relax1,
                                             yerr=df_errors1 / 100.,
                                             fmt='o', lw=1, color='#0d4f7a', alpha=0.5,
                                             label=label1)
    markers2, caps2, bars2 = pyplot.errorbar(ages2 / relax2,
                                             disk_fraction2 / 100.,
                                             xerr=ages_errors2 / relax2,
                                             yerr=df_errors2 / 100.,
                                             fmt='o', lw=1, color='#c28171', alpha=0.5,
                                             label=label2)

    [bar.set_alpha(0.5) for bar in bars1]
    [bar.set_alpha(0.5) for bar in bars2]

    # Plotting my data
    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    # 100 MSun
    all_fractions = []
    all_t_relax = []

    Rv = 0.5 | units.parsec
    g = 0.4

    for p in open_paths100:
        fractions = []
        t_relax = []
        print p

        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            if t == 0.:  # Have to do this to plot in terms of initial stellar mass, not considering accreted mass
                init_mass = stars.stellar_mass

            # Half mass relaxation time calculation
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum(), Rv)
            lr = stars.LagrangianRadii(unit_converter=converter, mf=[0.5])[0][0]  # Half mass radius
            bound = stars.bound_subset(tidal_radius=lr, unit_converter=converter)
            tdyn = numpy.sqrt(Rv ** 3 / (constants.G * bound.stellar_mass.sum()))
            N = len(bound)
            trh = 0.138 * (N / numpy.log(g * N)) * tdyn
            t_relax.append(1E-6 * trh.value_in(units.yr))

            stars.stellar_mass = init_mass
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.:
                print numpy.mean(stars[stars.bright == False].stellar_mass.value_in(units.MSun)), numpy.std(stars[stars.bright == False].stellar_mass.value_in(units.MSun))
                print stars[stars.bright == True].stellar_mass.value_in(units.MSun)
                print numpy.mean(stars[stars.bright == True].stellar_mass.value_in(units.MSun)), numpy.std(stars[stars.bright == True].stellar_mass.value_in(units.MSun))
                print 100. * (float(len(stars[stars.bright == True])) / float(len(stars)))

            fraction = float(len(disked_stars)) / float(len(small_stars))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_high = numpy.max(all_fractions, axis=0)
    disk_fractions_low = numpy.min(all_fractions, axis=0)
    disk_fractions_stdev = numpy.std(all_fractions, axis=0)

    pyplot.errorbar(times / numpy.mean(all_t_relax, axis=0),  # / (100. / (6 * numpy.log(100))),
                    all_disk_fractions,
                    #yerr=2 * disk_fractions_stdev,
                    color='k', lw=3,
                    label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')

    pyplot.fill_between(times / numpy.mean(all_t_relax, axis=0),#times / (100. / (6 * numpy.log(100))),
                        disk_fractions_high,
                        disk_fractions_low,
                        facecolor='black', alpha=0.2)

    # 50 MSun
    all_fractions = []
    all_t_relax = []

    for p in open_paths50:
        fractions = []
        t_relax = []
        print p

        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            if t == 0.:  # Have to do this to plot in terms of initial stellar mass, not considering accreted mass
                init_mass = stars.stellar_mass

            # Half mass relaxation time calculation
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum(), Rv)
            lr = stars.LagrangianRadii(unit_converter=converter, mf=[0.5])[0][0]
            bound = stars.bound_subset(tidal_radius=lr, unit_converter=converter)
            tdyn = numpy.sqrt(Rv ** 3 / (constants.G * bound.stellar_mass.sum()))
            N = len(bound)
            trh = 0.138 * (N / numpy.log(g * N)) * tdyn
            t_relax.append(1E-6 * trh.value_in(units.yr))

            stars.stellar_mass = init_mass
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.:
                print numpy.mean(stars[stars.bright == False].stellar_mass.value_in(units.MSun)), numpy.std(stars[stars.bright == False].stellar_mass.value_in(units.MSun))
                print stars[stars.bright == True].stellar_mass.value_in(units.MSun)
                print numpy.mean(stars[stars.bright == True].stellar_mass.value_in(units.MSun)), numpy.std(stars[stars.bright == True].stellar_mass.value_in(units.MSun))
                print 100. * (float(len(stars[stars.bright == True])) / float(len(stars)))

            fraction = float(len(disked_stars)) / float(len(small_stars))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_high = numpy.max(all_fractions, axis=0)
    disk_fractions_low = numpy.min(all_fractions, axis=0)
    disk_fractions_stdev = numpy.std(all_fractions, axis=0)

    pyplot.errorbar(times / numpy.mean(all_t_relax, axis=0),  # (30. / (6 * numpy.log(30))),
                    all_disk_fractions,
                    #yerr=2 * disk_fractions_stdev,
                    color='k',
                    ls='--', lw=3,
                    label=r'$\rho \sim 50 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')

    pyplot.fill_between(times / numpy.mean(all_t_relax, axis=0),
                        disk_fractions_high,
                        disk_fractions_low,
                        facecolor='black', alpha=0.2)

    pyplot.legend(framealpha=0.5)
    pyplot.xlabel("t / t$_\mathrm{relax}$ ")
    pyplot.ylabel("Disk fraction")
    pyplot.xlim([0.0, 3.0])
    pyplot.ylim([0.0, 1.0])

    if save:
        pyplot.savefig('{0}/disk_fraction.png'.format(save_path))

    pyplot.show()


def tests(open_path, i, N, t_end):
    times = numpy.arange(0.00, t_end + 0.05, 0.05)

    disk_sizes = []
    initial_size = 0.0

    mass_loss_pe, mass_loss_trunc = [], []
    g0s = []
    nearest_bright = []

    dispersal_time = 0.
    checked = False

    brighter = []
    brighter_masses = []

    for t in times:
        f = '{0}/N{1}_t{2}.hdf5'.format(open_path, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        small_stars = stars[stars.bright == False]
        s = stars[i]
        if not s.dispersed:
            print "t={0}, m={1}, disp={2}, enc={3}, size={4}, key={5}".format(t,
                                                           s.disk_mass.value_in(units.MJupiter),
                                                           s.dispersal_time.value_in(units.Myr),
                                                           s.encounters,
                                                           s.disk_radius.value_in(units.au),
                                                           s.key)


def disk_stellar_mass(open_paths100, open_paths50, t_end, mass_limit, save_path, save):
    fig = pyplot.figure()
    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    p = open_paths100[0]
    mass_limit = mass_limit | units.MSun
    initial_mass = 0.0

    low_mass_disks, high_mass_disks = [], []

    for p in open_paths100:
        low_all_in_p, high_all_in_p = [], []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:
                initial_mass = stars.stellar_mass
            stars.stellar_mass = initial_mass
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            low_all_in_p.append(float(len(low_mass_stars)))
            high_all_in_p.append(float(len(high_mass_stars)))

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.mean(low_mass_disks, axis=0)
    high_mass_disks = numpy.mean(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, label="low mass, 100 Ms".format(mass_limit.value_in(units.MSun)), lw=2)
    pyplot.plot(times, high_mass_disks, label=" high mass, 100 Ms".format(mass_limit.value_in(units.MSun)), lw=2)

    low_mass_disks, high_mass_disks = [], []

    for p in open_paths50:
        low_all_in_p, high_all_in_p = [], []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:
                initial_mass = stars.stellar_mass
            stars.stellar_mass = initial_mass
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            low_all_in_p.append(float(len(low_mass_stars)))
            high_all_in_p.append(float(len(high_mass_stars)))

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.mean(low_mass_disks, axis=0)
    high_mass_disks = numpy.mean(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, label="low mass, 50 Ms", ls=':', lw=2, color='orange')
    pyplot.plot(times, high_mass_disks, label="high mass, 50 Ms", ls=":", lw=2)
    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel('Disk fraction')
    pyplot.xlim([0.0, 5.0])
    pyplot.legend()
    if save:
        pyplot.savefig('{0}/stellar_mass.png'.format(save_path))

    pyplot.show()


def disk_stellar_mass_scatter(open_paths, N, t, save_path, save):
    fig = pyplot.figure()
    p = open_paths[0]
    mass_limit = 0.3 | units.MSun

    f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
    stars = io.read_set_from_file(f, 'hdf5', close_file=True)
    small_stars = stars[stars.bright == False]
    disked_stars = small_stars[small_stars.dispersed == False]

    high_mass_stars = disked_stars[disked_stars.stellar_mass >= mass_limit]
    low_mass_stars = disked_stars[disked_stars.stellar_mass < mass_limit]

    high_stellar_mass = high_mass_stars.stellar_mass.value_in(units.MSun)
    high_disk_mass = high_mass_stars.disk_mass.value_in(units.MJupiter)

    low_stellar_mass = low_mass_stars.stellar_mass.value_in(units.MSun)
    low_disk_mass = low_mass_stars.disk_mass.value_in(units.MJupiter)

    pyplot.scatter(high_stellar_mass, high_disk_mass, color='red', label='high mass')
    pyplot.scatter(low_stellar_mass, low_disk_mass, color='blue', label='low mass')
    pyplot.xlabel('Stellar mass [$M_\odot$]')
    pyplot.ylabel('Disk mass [$M_{Jup}$]')
    pyplot.legend()
    pyplot.show()


def g0_in_time(open_paths100, open_paths30, save_path, N, i):
    #fig = pyplot.figure()
    #ax = pyplot.gca()

    times = numpy.arange(0.0, 5.05, 0.05)

    g0s100, g0s30 = [], []
    g0s100_low, g0s100_high = [], []
    g0s30_low, g0s30_high = [], []

    for t in times:
        g0_in_time = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.bright == False]
            g0 = small_stars.g0

            g0_in_time.append(numpy.mean(g0))

        g0s100.append(numpy.mean(g0_in_time))
        g0s100_low.append(numpy.min(g0_in_time))
        g0s100_high.append(numpy.max(g0_in_time))

    for t in times:
        g0_in_time = []
        for p in open_paths30:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.bright == False]
            g0 = small_stars.g0

            g0_in_time.append(numpy.mean(g0))

        g0s30.append(numpy.mean(g0_in_time))
        g0s30_low.append(numpy.min(g0_in_time))
        g0s30_high.append(numpy.max(g0_in_time))

    print numpy.mean(g0s100), numpy.min(g0s100_low[g0s100_low > 0]), numpy.max(g0s100_high)
    print numpy.mean(g0s30), numpy.min(g0s30_low[g0s30_low > 0]), numpy.max(g0s30_high)

    """ax.semilogy(times, g0s100, lw=3, color='black',
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')
    ax.fill_between(times,
                   g0s100_low,
                   g0s100_high,
                   facecolor='black', alpha=0.2)
    ax.semilogy(times, g0s30, lw=3, ls='--', color='black',
                label=r'$\rho \sim 30 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')
    ax.fill_between(times,
                   g0s30_low,
                   g0s30_high,
                   facecolor='black', alpha=0.2)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$\mathrm{G}_0 \ [\mathrm{erg}/\mathrm{cm}^2$]')
    ax.set_xlim([0.05, 5.0])
    ax.legend(loc='upper right', framealpha=0.4)
    if save:
        pyplot.savefig('{0}/g0.png'.format(save_path))
    pyplot.show()"""


def brightstars(open_paths100, open_paths50):
    dt = 0.5  #| units.Myr
    t_end = 5.0   # | units.Myr
    times = numpy.arange(0.0, t_end + dt, dt)

    colors = ["#638ccc",
              "#ca5670",
              "#72a555",
              "#ab62c0",
              "#c57c3c",
              "black"]

    for p in open_paths100:
        all_dists = []
        all_t = []

        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        small_stars = stars[stars.bright == False]
        init_bright_stars = stars[stars.bright == True]
        #disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
        disked_stars = small_stars[small_stars.dispersed == False]

        # This is super slow but it's the only way my brain actually sees the steps in the correct order
        for i in range(len(init_bright_stars)):
            all_bright_i = []
            for t in times:
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                small_stars = stars[stars.bright == False]
                bright_stars = stars[stars.bright == True]
                # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
                disked_stars = small_stars[small_stars.dispersed == False]

                distances = []

                # All distances from bright star i to all disked stars at time t
                for d in disked_stars:
                    dist = distance(bright_stars[i], d).value_in(units.parsec)
                    distances.append(dist)

                all_bright_i.append(distances)

            means = []
            for b in all_bright_i:
                means.append(numpy.mean(b))

            pyplot.plot(times, means, color=colors[open_paths100.index(p)])


    for p in open_paths50:
        all_dists = []
        all_t = []

        f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        small_stars = stars[stars.bright == False]
        init_bright_stars = stars[stars.bright == True]
        #disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
        disked_stars = small_stars[small_stars.dispersed == False]

        # This is super slow but it's the only way my brain actually sees the steps in the correct order
        for i in range(len(init_bright_stars)):
            all_bright_i = []
            for t in times:
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                small_stars = stars[stars.bright == False]
                bright_stars = stars[stars.bright == True]
                # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
                disked_stars = small_stars[small_stars.dispersed == False]

                distances = []

                # All distances from bright star i to all disked stars at time t
                for d in disked_stars:
                    dist = distance(bright_stars[i], d).value_in(units.parsec)
                    distances.append(dist)

                all_bright_i.append(distances)

            means = []
            for b in all_bright_i:
                means.append(numpy.mean(b))

            pyplot.plot(times, means, color=colors[3 + open_paths50.index(p)])

    pyplot.show()


def deltat(open_paths100, open_paths50, save_path, mass_limit=0.0, save=False):

    colors = ["#638ccc",
              "#ca5670",
              "#72a555",
              "#ab62c0",
              "#c57c3c",
              "black"]

    dt = 0.5  #| units.Myr
    t_end = 5.0   # | units.Myr
    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    for p in open_paths100:
        prev_mdot = 0.0
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.0:
                init_mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)
                prev_mdot = len(stars)

            if t == 0.1:
                print numpy.mean(10. * small_stars[small_stars.dispersed == True].initial_disk_mass.value_in(units.MSun))
                print small_stars[small_stars.dispersed == True].initial_disk_size.value_in(units.au)

            # Mass lost due to photoevaporation in dt
            #mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)# - init_mdot
            init_mdot = mdot
            #prev_mdot = len(mdot)

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)) / float(prev_mdot))  # Number of disks that lost more than 5MJup in t

        cumulative = numpy.array(all_mdot).cumsum()

        pyplot.plot(times, cumulative / cumulative[-1], lw=3, c='navy')

    for p in open_paths50:
        prev_mdot = 0.0
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]
            #print t
            #print disked_stars.g0
            #print disked_stars.photoevap_mass_loss.value_in(units.MJupiter)
            #print disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            if t == 0.0:
                init_mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)
                prev_mdot = len(stars)

            if t == 0.1:
                print numpy.mean(10. * small_stars[small_stars.dispersed == True].initial_disk_mass.value_in(units.MSun))
                print small_stars[small_stars.dispersed == True].initial_disk_size.value_in(units.au)

            # Mass lost due to photoevaporation in dt
            #mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)# - init_mdot
            #prev_mdot = mdot

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)) / float(prev_mdot))  # Number of disks that lost more than 5MJup in t

            #print mdot

            #pyplot.plot(times, all_mdot, 'o', alpha=0.2, color='k')
            #pyplot.plot(t * numpy.ones((len(disked_stars))), mdot/dt)

        #print all_mdot
        cumulative = numpy.array(all_mdot).cumsum()
        pyplot.plot(times, cumulative / cumulative[-1], '--', lw=3, c='navy')
        #cumulative_mdot_5MJup)

    pyplot.xlabel(r'$t$ [Myr]', fontsize=28)
    pyplot.ylabel(r'$f_{\dot{M}(t) > 5 \mathrm{\ M_{Jup}}}$', fontsize=30)
    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0.0, 1.0])
    #ax2.set_xticks(times)
    pyplot.grid(True, alpha=0.2)

    # Custom legend
    custom_lines = [mlines.Line2D([0], [0], color='navy', lw=3),
                    mlines.Line2D([0], [0], color='navy', lw=3, ls='--')]

    pyplot.legend(custom_lines, [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                              r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
                  loc='lower right', framealpha=0.4)

    if save:
        pyplot.savefig('{0}/mass_loss_cumulative.png'.format(save_path))
    pyplot.show()


def cumulative_mass(open_paths100, open_paths50):
    #f, (ax1, ax2) = pyplot.subplots(1, 2)#, figsize=(10, 15), sharex=True)

    times = [0.0, 0.05, 2.5, 5.0]
    colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

    i = 0
    open_paths100 = [open_paths100[0]]
    open_paths50 = [open_paths50[1]]

    for t in times:
        #all_pp = []
        all_masses = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]
            disks_mass = disked_stars.disk_mass.value_in(units.MJupiter)

            sorted_disks_mass = numpy.sort(disks_mass)
            #pp = 1. * numpy.arange(len(sorted_disks_mass)) / (len(stars) - 1)

            #all_pp.append(pp)
            all_masses.append(sorted_disks_mass)

            cumulative = 1. * numpy.arange(len(sorted_disks_mass)) / (len(stars) - 1)

            pyplot.plot(sorted_disks_mass, cumulative, c=colors[2 + i], lw=2, label='{0} Myr'.format(t))

        i += 1

        '''try:
            disk_masses = numpy.mean(all_masses, axis=0)
            disk_masses_stdev = numpy.std(all_masses, axis=0)
        except ValueError:
            max_len = 0
            for a in all_masses:
                if len(a) > max_len:
                    max_len = len(a)

            new_sorted = []
            for a in all_masses:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant')
                new_sorted.append(b)

            disk_masses = numpy.mean(new_sorted, axis=0)
            disk_masses_stdev = numpy.std(new_sorted, axis=0)'''

    for p in open_paths50:
        i = 0
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]
            disks_mass = disked_stars.disk_mass.value_in(units.MJupiter)
            sorted_disks_mass = numpy.sort(disks_mass)

            pp = 1. * numpy.arange(len(sorted_disks_mass)) / (len(stars) - 1)
            pyplot.plot(sorted_disks_mass, pp, '--', c=colors[2 + i], lw=2, label='{0} Myr'.format(t))
            i += 1

    pyplot.xlim([0.0, 140])
    pyplot.ylim([0.0, 1.0])
    pyplot.xlabel(r'Disk mass $[\mathrm{M_{Jup}}]$')
    pyplot.ylabel(r'$f$')

    # Custom legend
    #custom_lines = [[mlines.Line2D([0], [0], color='navy', lw=3),
    #                mlines.Line2D([0], [0], color='navy', lw=3, ls='--')]]

    #pyplot.legend(custom_lines, [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
    #                          r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
    #              loc='lower right', framealpha=0.4)

    """ax2.legend(loc='lower right')
    ax2.set_xlim([0.0, 150])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel(r'Disk mass $[\mathrm{M_{Jup}}]$')
    ax2.set_ylabel(r'$f$')"""
    pyplot.legend()
    pyplot.show()


def mdot(open_paths100, open_paths50):

    tend = 5.0
    times = numpy.arange(0.0, tend + 0.05, 0.05)

    cm = pyplot.cm.get_cmap('RdYlBu')

    for p in open_paths100:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        init_masses = stars.stellar_mass.value_in(units.MSun)
        #stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
        small_stars = stars[stars.bright == False]
        #small_stars = small_stars[small_stars.dispersed == False]
        prev_stars = stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

        N = len(small_stars)
        max_mdots = []
        max_times = []
        star_mass = []

        for i in range(N):
            all_mdot = []
            prev_mdot = prev_stars[i]
            for t in times[1:]:
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                #stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
                small_stars = stars[stars.bright == False]
                #small_stars = small_stars[small_stars.dispersed == False]
                #print i, len(small_stars)
                #print small_stars[i]

                all_mdot.append(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot)
                prev_mdot = small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)
                #print len(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)), len(prev_stars)
                #prev_stars = stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            m = max(all_mdot)
            max_mdots.append(float(m) / 50000)
            max_times.append(times[all_mdot.index(m)])
            star_mass.append(init_masses[i])

        print max_mdots
        pyplot.scatter(max_times, max_mdots,
                       c=star_mass, cmap=cm,
                       alpha=0.4)

    '''for i in range(N):
        all_mdot = []
        prev_mdot = prev_stars[i]
        for t in times[1:]:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            small_stars = stars[stars.bright == False]
            # small_stars = small_stars[small_stars.dispersed == False]
            # print i, len(small_stars)
            # print small_stars[i]

            all_mdot.append(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot)
            prev_mdot = small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)
            # print len(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)), len(prev_stars)
            # prev_stars = stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

        m = max(all_mdot)
        max_mdots.append(float(m) / 0.05)
        max_times.append(times[all_mdot.index(m)])

    print max_mdots
    pyplot.scatter(max_times, max_mdots, alpha=0.4, c='k')'''

    pyplot.xlabel(r'Time [Myr]')
    pyplot.ylabel(r'$\max(\d\dot{M} / dt)$')
    pyplot.show()


def triplefigure(open_paths100, open_paths50, save_path, mass_limit=0.0, save=False):
    f, (ax2, ax3) = pyplot.subplots(2, 1, figsize=(10, 15), sharex=True)
    colors = ["#638ccc",
              "#ca5670",
              "#72a555",
              "#ab62c0",
              "#c57c3c",
              "black"]

    tend = 5.0
    times = numpy.arange(0.0, tend + 0.05, 0.05)

    """photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []
    disk_mass, mass_low, mass_high = [], [], []

    for t in times:
        photoevap_in_t, trunc_in_t = [], []
        mass_in_t = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            small_stars = stars[stars.bright == False]
            #small_stars = small_stars1[small_stars1.dispersed == False]

            photoevap_in_t.append(numpy.mean(small_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)))
            trunc_in_t.append(numpy.mean(small_stars.cumulative_truncation_mass_loss.value_in(units.MJupiter)))
            mass_in_t.append(numpy.mean(small_stars.disk_mass.value_in(units.MJupiter)))

            #if t == 0.05:
            #    for s in small_stars1:
            #        print s.initial_disk_mass.in_(units.MSun), s.dispersed
            #    print "*"

        photoevap_mass_loss.append(numpy.mean(photoevap_in_t))
        photoevap_low.append(numpy.min(photoevap_in_t))
        photoevap_high.append(numpy.max(photoevap_in_t))

        trunc_mass_loss.append(numpy.mean(trunc_in_t))
        trunc_low.append(numpy.min(trunc_in_t))
        trunc_high.append(numpy.max(trunc_in_t))

        disk_mass.append(numpy.mean(mass_in_t))
        mass_low.append(numpy.min(mass_in_t))
        mass_high.append(numpy.max(mass_in_t))

    ax1.plot(times, photoevap_mass_loss, label="Photoevaporation", lw=3, color="#009bed")
    ax1.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    facecolor="#009bed",
                    alpha=0.2)

    ax1.plot(times, trunc_mass_loss, label="Dynamical truncations", lw=3, color="#d73027")
    ax1.fill_between(times,
                    trunc_low,
                    trunc_high,
                    facecolor="#d73027",
                    alpha=0.2)

    photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []
    disk_mass, mass_low, mass_high = [], [], []

    for t in times:
        photoevap_in_t, trunc_in_t = [], []
        mass_in_t = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            stars = stars[stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            small_stars = stars[stars.bright == False]

            photoevap_in_t.append(numpy.mean(small_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)))
            trunc_in_t.append(numpy.mean(small_stars.cumulative_truncation_mass_loss.value_in(units.MJupiter)))
            mass_in_t.append(numpy.mean(small_stars.disk_mass.value_in(units.MJupiter)))

        photoevap_mass_loss.append(numpy.mean(photoevap_in_t))
        photoevap_low.append(numpy.min(photoevap_in_t))
        photoevap_high.append(numpy.max(photoevap_in_t))

        trunc_mass_loss.append(numpy.mean(trunc_in_t))
        trunc_low.append(numpy.min(trunc_in_t))
        trunc_high.append(numpy.max(trunc_in_t))

        disk_mass.append(numpy.mean(mass_in_t))
        mass_low.append(numpy.min(mass_in_t))
        mass_high.append(numpy.max(mass_in_t))

    ax1.plot(times, photoevap_mass_loss, label="Photoevaporation", ls="--", lw=3, color="#009bed")
    ax1.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    alpha=0.2, facecolor="#009bed", edgecolor='#009bed', hatch="/")

    ax1.plot(times, trunc_mass_loss, label="Dynamical truncations", ls="--", lw=3, color="#d73027")
    ax1.fill_between(times,
                    trunc_low,
                    trunc_high,
                    alpha=0.2, facecolor="#d73027", edgecolor='#d73027', hatch="/")

    ax1.set_xlim([0.0, 5.0])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    ax1.legend([PhotoevapObject(), TruncationObject(), M100shadedObject, M30shadedObject],
               ['Photoevaporation', 'Dynamical truncations',
                r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
               handler_map={PhotoevapObject: PhotoevapObjectHandler(),
                            TruncationObject: TruncationObjectHandler(),
                            M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
               loc='best', bbox_to_anchor=(0.85, -0.15), ncol=2,
              fontsize=20, framealpha=1.)
    ax1.tick_params(which='minor', direction='out', length=6, width=1)
    ax1.tick_params(which='major', direction='out', length=6, width=1)
    #pyplot.xticks([0.1, 1, 5], ['0.1', '1', '5'])

    #pyplot.tight_layout()"""

    times = numpy.arange(0.0, tend + 0.05, 0.05)

    for p in open_paths100:
        prev_mdot = 0.0
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            #disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.0:
                init_mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)
                prev_mdot = len(init_mdot)

            if t == 0.05:
                print small_stars[small_stars.dispersed == True].initial_disk_mass.in_(units.MJupiter)

            # Mass lost due to photoevaporation in dt
            #mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)# - init_mdot
            init_mdot = mdot
            prev_mdot = len(mdot)

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)) / float(len(small_stars)))  # Number of disks that lost more than 5MJup in t

        cumulative = numpy.array(all_mdot).cumsum()

        ax2.plot(times, cumulative / cumulative[-1], lw=2, c=colors[open_paths100.index(p)])

    for p in open_paths50:
        prev_mdot = 0.0
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            #disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]
            #print t
            #print disked_stars.g0
            #print disked_stars.photoevap_mass_loss.value_in(units.MJupiter)
            #print disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            if t == 0.0:
                init_mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)
                #prev_mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            if t == 0.05:
                print small_stars[small_stars.dispersed == True].initial_disk_mass.in_(units.MJupiter)

            # Mass lost due to photoevaporation in dt
            #mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)# - init_mdot
            #prev_mdot = mdot

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)) / float(len(small_stars)))  # Number of disks that lost more than 5MJup in t

            #print mdot

            #pyplot.plot(times, all_mdot, 'o', alpha=0.2, color='k')
            #pyplot.plot(t * numpy.ones((len(disked_stars))), mdot/dt)

        #print all_mdot
        cumulative = numpy.array(all_mdot).cumsum()
        ax2.plot(times, cumulative / cumulative[-1], ':', lw=2, c=colors[3 + open_paths50.index(p)])
        #cumulative_mdot_5MJup)

    ax2.set_xlabel(r'$\delta_t$ [Myr]', fontsize=28)
    ax2.set_ylabel(r'$\dot{M}(\delta_t) > 5 \mathrm{\ M_{Jup}}$', fontsize=30)
    ax2.set_xlim([0.0, 5.0])
    ax2.set_ylim([0.0, 1.0])
    #ax2.set_xticks(times)
    ax2.grid(True, alpha=0.2)

    # Custom legend
    custom_lines = [mlines.Line2D([0], [0], color='navy', lw=2),
                    mlines.Line2D([0], [0], color='navy', lw=2, ls=':')]

    ax2.legend(custom_lines, [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                                 r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
                  loc='lower right', framealpha=0.4)

    for p in open_paths100:
        all_dists = []
        all_t = []

        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        small_stars = stars[stars.bright == False]
        init_bright_stars = stars[stars.bright == True]
        # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
        disked_stars = small_stars[small_stars.dispersed == False]

        # This is super slow but it's the only way my brain actually sees the steps in the correct order
        for i in range(len(init_bright_stars)):
            all_bright_i = []
            for t in times:
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                small_stars = stars[stars.bright == False]
                bright_stars = stars[stars.bright == True]
                # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
                disked_stars = small_stars[small_stars.dispersed == False]

                distances = []

                # All distances from bright star i to all disked stars at time t
                for d in disked_stars:
                    dist = distance(bright_stars[i], d).value_in(units.parsec)
                    distances.append(dist)

                all_bright_i.append(distances)

            means = []
            for b in all_bright_i:
                means.append(numpy.mean(b))

            ax3.plot(times, means, color=colors[open_paths100.index(p)])

    for p in open_paths50:
        all_dists = []
        all_t = []

        f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        small_stars = stars[stars.bright == False]
        init_bright_stars = stars[stars.bright == True]
        # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
        disked_stars = small_stars[small_stars.dispersed == False]

        # This is super slow but it's the only way my brain actually sees the steps in the correct order
        for i in range(len(init_bright_stars)):
            all_bright_i = []
            for t in times:
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                small_stars = stars[stars.bright == False]
                bright_stars = stars[stars.bright == True]
                # disked_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
                disked_stars = small_stars[small_stars.dispersed == False]

                #if open_paths50.index(p) == 1 and i == 0:
                #    print small_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

                distances = []

                # All distances from bright star i to all disked stars at time t
                for d in disked_stars:
                    dist = distance(bright_stars[i], d).value_in(units.parsec)
                    distances.append(dist)

                all_bright_i.append(distances)

            means = []
            for b in all_bright_i:
                means.append(numpy.mean(b))

            ax3.plot(times, means, color=colors[3 + open_paths50.index(p)])

    pyplot.show()

    if save:
        pyplot.savefig('{0}/mass_loss.png'.format(save_path))
    pyplot.show()


def test(files, N):
    times = numpy.arange(0.0, 5.0 + 0.05, 0.05)
    i = 0
    for p in files:
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            bright_stars = stars[stars.bright == True]
            disked_stars = small_stars[small_stars.dispersed == False]

            pyplot.clf()
            pyplot.scatter(disked_stars.x.value_in(units.parsec),
                           disked_stars.y.value_in(units.parsec),
                           c='blue')
            pyplot.scatter(bright_stars.x.value_in(units.parsec),
                           bright_stars.y.value_in(units.parsec),
                           c='red', s=bright_stars.stellar_mass.value_in(units.MSun))

            pyplot.savefig('./test_plots4/{0}.png'.format(i))
            i += 1


def main(save_path, time, N, distribution, i, all_distances, single, save):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    paths100 = ['results/final/plummer_N100_1/',
                'results/final/plummer_N100_2/',
                'results/final/plummer_N100_3/']

    paths50 = ['results/c50/plummer_N50_1/0/',
               'results/c50/plummer_N50_4/0/',
               'results/c50/plummer_N50_3/0/']#,
               #'results/c50/plummer_N50_4/0/']

    path = 'results/final/plummer_N100_1/'

    #for j in range(100):
    #    print j
    #    single_star(path, save_path + '/single_N100_1', N, j, time, all_distances)

    if single:
        single_star(path, save_path, N, i, time, all_distances)
        #tests(path, i, N, time)
    else:
        #mass_loss_in_time(paths100, paths50, save_path, time, save=True, mass_limit=0.0)

        #brightstars(paths100, paths50)

        #triplefigure(paths100, paths50, save_path)

        #test(paths50, 50)

        #disk_fractions(paths100, paths50, time, save_path, save=False, mass_limit=0.3)

        #deltat(paths100, paths50, save_path, save=True)

        #cumulative_mass(paths100, paths50)

        mdot(paths100, paths50)

        #disk_mass(paths100, paths50, save_path, time, save)
        #disk_size(paths100, paths50, save_path, time, save)
        #disk_stellar_mass(paths100, paths50, time, 0.1, save_path, save)
        #disk_stellar_mass_scatter(paths, N, time, save_path, save)
        #luminosity_vs_mass(save_path, save)
        #g0_in_time(paths100, paths50, save_path, 100, 0)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    result.add_option("-S", dest="save", type="int", default=0,
                      help="save plot? [%default]")

    # Simulation parameters
    result.add_option("-s", dest="save_path", type="string", default='/media/fran/data1/photevap-paper/figures',
                      help="path to save the results [%default]")
    result.add_option("-t", dest="time", type="float", default='5.0',
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
    result.add_option("-i", dest="i", type="int", default=0,
                      help="Individual star to plot [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

