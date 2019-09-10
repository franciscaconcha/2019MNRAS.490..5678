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


def cumulative_mass(open_paths100, open_paths50, save_path):
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
    pyplot.savefig('{0}/cdf_disk_mass.png'.format(save_path))
    pyplot.show()


def mdot(open_paths100, open_paths50, save_path):

    tend = 5.0
    times = numpy.arange(0.0, tend + 0.05, 0.05)
    #times = numpy.arange(0.0, tend + 0.5, 1.5)

    cm = pyplot.cm.get_cmap('RdYlBu')

    '''for p in open_paths100:
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
        print max_times
        print star_mass'''

    mdots = [[6.146843956410559e-05, 0.00014138128297285215, 0.00011295935097482571, 4.921713011791867e-07,
              0.0002854939332551815, 0.00033448059408899974, 0.0004184428646798976, 0.00014027765126241182,
              0.00011679707854370779, 0.000558470676574507, 0.0012557797510657736, 0.00028073758687101373,
              5.974584000997858e-05, 0.0001396008315046056, 9.137915562986199e-05, 0.000688980100621381,
              0.0005125585397427104, 0.00010980448212129949, 0.0007819567239386667, 0.00014113519732226248,
              0.0001395766548685285, 0.00027903026691176216, 0.00017915549147973925, 0.0002255085591986688,
              0.0006988965699462788, 0.0015633337995465172, 0.0009125728395429931, 0.0007900592284222077,
              0.00014101215449696786, 0.0001409301259467712, 0.0004185659075051924, 0.0002799735952390223,
              0.0005049079545863153, 3.281142007861245e-07, 5.978685428507685e-05, 1.6405710039306224e-07,
              0.0001128363081495309, 0.0002258366733994549, 0.0002697404419420115, 0.00012213936140550814,
              7.391679707359632e-05, 0.0006171314758339274, 0.0004780035507669247, 7.391904276542179e-05,
              0.00023898248185819077, 8.528153317187727e-05, 1.1894139778497004e-06, 6.171452521469517e-05,
              7.392206965083824e-05, 0.00019103944637039117, 0.00019926553142155591, 8.012373265266834e-05,
              0.0002795381331698714, 0.0004265462920126526, 4.77059388078344e-07, 1.3357642923334865e-05,
              0.00011312038118980231, 6.9835657717972435e-06, 0.00011191241714151524, 0.0008900297998273999,
              0.00013949462631833196, 0.0007245497254189388, 3.6610159046793247e-07, 4.7815155471660376e-06,
              0.000409835958066612, 0.0007381975038078299, 0.00010718543734045752, 3.4098154089752654e-07,
              5.974056743273667e-05, 3.797290627264093e-05, 6.322370183573306e-05, 0.0003744193861872527,
              0.0001497039621926108, 0.00011497414563430252, 2.050713754913347e-06, 5.2148455230469575e-05,
              0.00030631865235490976, 5.6059424759613726e-05, 0.0004625014842637173, 2.2651694234366816e-06,
              4.302131978635842e-05, 0.0001370724912666646, 5.974584000997858e-05, 0.00029008564714451895,
              0.00024078510921405426, 0.00023468484282178595, 0.0001931511513690191, 7.580872630535831e-05,
              2.2698535786977485e-06, 6.171149832927873e-05, 4.781515547166073e-06, 2.5696768718674184e-05,
              2.050713754913347e-06, 6.361144464326303e-05, 0.00010330903645036086, 4.781515547166499e-06,
              4.781515547165594e-06],
             [0.00022955135814304217, 0.0003945707560752965, 0.00011679707854370779, 7.387154499758496e-05,
              0.00011191241714151524, 0.000213359732512219, 0.0002995345883439552, 9.937573426254037e-05,
              0.00020801744965041866, 0.000408182786959744, 0.00031686030231417925, 0.0002794422938660702,
              9.129288927875237e-05, 9.937573426254037e-05, 2.5576707436996936e-05, 0.0001610946141792991,
              0.00019403965353521934, 0.00013833098703633132, 0.0010410192799708616, 0.0003325227348620562,
              0.0009458561740760578, 0.00016509942573399072, 9.937573426254037e-05, 0.00021213001386187475,
              0.0005063123383535473, 0.0006292752748865006, 9.129288927875237e-05, 2.5576707436996936e-05,
              0.00023072557230037315, 0.0007246082102046675, 2.5576707436996936e-05, 0.00022523208992524386,
              0.0002072453524596822, 5.974056743273667e-05, 0.0007681897811035431, 0.0005615140055356179,
              0.0003123385550853611, 9.937573426254037e-05, 0.0001394441016650787, 9.937573426254037e-05,
              9.449107286034781e-05, 0.0002246666967408496, 0.0001483894992593492, 0.0001866846784336775,
              0.00046047568156860506, 0.00039627285464420494, 0.0002843715224757299, 0.0002992378207571295,
              0.0006033815502162371, 0.0004520319435525724, 0.0002031327882482261, 0.00039584321534869536,
              0.0001250783251160972, 0.00023468484282178608, 0.0003579914758499899, 2.5576707436996936e-05,
              0.00029008564714451895, 0.0001554578088456086, 0.0005387136349697094, 0.0003224634470622707,
              0.0007048401541878724, 0.00025616866203674496, 0.0006053859012541125, 0.0002984657235663931,
              0.00019958427548602672, 0.0001125397849291807, 0.00023468484282178608, 2.5576707436996936e-05,
              7.387154499758496e-05, 7.387154499758496e-05, 2.5576707436996936e-05, 0.00016954976424444717,
              0.0002670512819964906, 0.00037220611461339373, 0.0005563391431737874, 0.00013861827808048054,
              7.383476852339978e-05, 0.0002103199478988884, 0.0011356484668429, 6.013422690971793e-05,
              0.00011679707854370779, 0.00018624199406334455, 1.838823709258337e-06, 6.433082235801605e-05,
              0.0003807722304936185, 0.00013940957088271898, 1.6060577924874357e-05, 7.387905462727985e-05,
              8.133529428429721e-05, 0.000116523947218976, 0.00015847875428261083, 0.00029008564714451895,
              0.0001246001067792566, 0.00032060093714500785, 0.0003585646250838437, 9.12561128045672e-05,
              0.0001576608108503348, 0.00015360404121685832, 1.0791196414264707e-05],
             [5.974584000997858e-05, 0.0013454797298960584, 0.001563732393880581, 0.00036892294163607927,
              0.0013942347710323838, 0.00011679707854370779, 0.00018128576047864013, 0.0009056437167944951,
              0.00016510142648245083, 4.457861600556265e-06, 0.00028070882530537445, 0.00014105316877206607,
              4.8347650578751654e-05, 0.0001215809347437598, 0.00026772227538529895, 9.937573426254037e-05,
              0.0002255905877488653, 0.0011565758408399187, 0.000320771729249753, 7.388963360248092e-05,
              9.937573426254037e-05, 0.0004921102154970803, 0.00018260479931381476, 0.0003179350613767411,
              0.0008234534426362905, 2.169934970850344e-05, 0.0010270163908388341, 0.001003398022314348,
              0.00018253966175324891, 0.0003004692245577527, 0.00018771494495841836, 0.0002737841561454274,
              2.5576707436996936e-05, 0.00036909296817812676, 0.002303107313703576, 7.865779324060913e-05,
              0.0006769029805980127, 0.0009257073313483879, 0.0001221423882909246, 4.8347650578751654e-05,
              0.0014248364853058954, 1.6103021497660468e-07, 0.0002818164764759621, 0.0006750476133098757,
              0.0003668078136213257, 0.00025241296301805943, 0.0008793805021538047, 0.0005353248781729134,
              0.0001651424407575491, 9.937573426254037e-05, 0.00019926855830697235, 2.050713754913347e-06,
              8.528153317187727e-05, 0.00011191241714151524, 0.0002072453524596822, 0.00046729325059433215,
              0.00034527390512870235, 0.0004192292168941878, 0.0003934251175150286, 0.00020891885370647066,
              0.00028906993360011733, 0.0005588455935350333, 2.050713754913347e-06, 0.0002072453524596822,
              2.926094669904594e-05, 0.0003219655940984675, 0.0002325606398044646, 0.00027693181507376955,
              0.00018824944508514662, 2.1708089292861922e-06, 0.00029008564714451895, 9.13403870465892e-05,
              0.0002625163169217196, 0.00022523208992524386, 0.00023468484282178437, 9.3440237141679e-05,
              0.0004875549129458425, 0.0001221423882909246, 4.781515547165922e-06, 2.5576707436996936e-05,
              9.323007058652572e-05, 0.00015414270623411217, 7.580872630535831e-05, 2.050713754913347e-06,
              5.974584000997858e-05, 0.00021331133633164886, 8.092593065132905e-06, 5.974296837461622e-05,
              9.449107286034781e-05, 5.031204667009778e-05, 2.3441548434648845e-05, 0.00011191241714151524,
              0.0002816456843712169, 0.00018986453136320407, 9.666563550684724e-05, 0.0001356208882806382],
             [0.12267492192337287, 0.006995238453465731, 0.011553764186250164, 0.03487874927061946, 0.01124451879004858, 0.00010253568774566401, 0.00023907577735831075, 0.09069406495821984, 0.06845744149837983, 0.06277258184324937, 0.013027870231295632, 0.004322241220552032, 0.009291724339617674, 0.012624749578412796, 0.031777155260192196, 0.006164826058835837, 0.0042914469305737135, 0.003066815688584037, 0.005647967548741286, 0.026582817665893797, 0.025139250056220558, 0.004972888140636846, 0.001129488625600743, 0.021167702729265638, 0.006721199294009522, 0.00010253568774566401, 0.00010253568774567511, 0.028762624517546364, 0.010211036817831741, 0.007071114862397531, 0.00010253568774566401, 0.018262463558379516, 0.00010253568774566401, 0.014115643713236287, 0.005711846737702678, 0.031160978609261532, 0.03684635311509914, 0.006570154273283512, 0.00028731123350524514, 0.010242434807979794, 0.007552411983344298, 0.011349469654528559, 0.01036841976424885, 0.005750546105424369, 0.04169274187912746, 0.00010253568774566401, 0.0001132584711718343],
             [0.00702190506222035, 9.194118546291818e-05, 0.007083335547783847, 0.0032363374666507165, 0.003723860771567776, 0.009815913252971065, 0.0014889842285479382, 0.01971598481770795, 0.005710223056882069, 0.0028005701174232734, 0.003112127020042133, 0.029347740503915396, 0.004571999758774652, 0.00023907577735831075, 0.00023907577735829966, 0.0004134577287703669, 0.007169402827632821, 0.007051326241568485, 0.0006210960759765937, 0.03368524752881572, 0.00023907577735831075, 0.019576058864607438, 0.028295550326764052, 0.014018444113605533, 0.006116531101369741, 0.00029625604044428866, 0.00023024770184457678, 0.007575330703986955, 0.009780061379425801, 0.03368524752881571, 0.00023907577735831075, 0.005846422527625373, 0.0031402120285577402, 0.0011462765871359689, 0.0006210960759765937, 0.00039457911306977147, 0.00023907577735831075, 0.03093962061634661, 0.00023907577735831075, 9.194118546291818e-05, 0.0009789932084504163, 0.0004809000681632438, 0.00039457911306977147, 0.027550529394655215, 0.0011462765871359689, 0.003719372520625306, 0.0025865099035423354, 0.017521278564780557],
             [0.001143282156481617, 0.0036955283581797797, 0.018332337759386568, 0.001173975209278873, 0.0058113354797575695, 0.0021284332892416878, 0.002800570117423274, 0.0028005701174232734, 9.194118546291818e-05, 0.03237647000919343, 0.00023907577735829966, 0.0075063007323061355, 0.007155058280991323, 0.006164818296333132, 0.00023907577735831075, 0.029371518268815393, 0.008794958349848898, 0.0733583182168099, 0.007933911421232419, 0.0070231254400610925, 0.004972412621729095, 0.08392191707077855, 0.006933703080831897, 0.015118214346787763, 0.0049673461697836395, 0.007178071503612934, 0.0016909157883802726, 0.00023907577735829966, 0.0013907886807480745, 0.039106851567502335, 0.0030227743585847164, 0.0007649563937081219, 0.0017685407031692293, 0.03582040623391508, 0.009505039294257547, 0.0021607058547020905, 0.00629471169109695, 0.0008136860258950569, 0.00023907577735831075, 0.00023907577735831075, 0.00456659557223815, 0.0006445900193979645, 0.0005211986638375192, 0.022911464653855844, 0.0022433724624392314, 0.0034858283033807895, 0.031231535092982547, 0.006921526034509239, 0.002081900466134547]]

    max_times = [[2.0, 0.80000000000000004, 0.0, 0.0, 0.0, 0.0, 1.3, '0.55000000000000004', '0.0', '0.0', '0.60000000000000009', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.45000000000000001', '0.0', '2.2000000000000002', '0.0', '0.0', '0.0', '0.75', '0.60000000000000009', '0.0', '0.0', '1.05', '0.35000000000000003', '0.35000000000000003', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.1000000000000001', '0.0', '1.0', '0.0', '1.3500000000000001', '0.0', '0.0', '0.0', '4.9500000000000002', '0.0', '0.0', '0.0', '3.0', '0.0', '1.6500000000000001', '0.0', '1.3500000000000001', '0.0', '0.0', '3.1000000000000001', '0.0', '0.0', '4.0499999999999998', '4.7000000000000002', '0.0', '0.0', '0.0', '0.55000000000000004', '2.0', '0.0', '1.0', '0.0', '0.60000000000000009', '0.35000000000000003', '1.2000000000000002', '0.0', '1.3', '0.0', '3.3000000000000003', '0.0', '0.0', '0.0', '0.0', '4.9500000000000002', '2.1000000000000001', '0.5', '0.65000000000000002', '0.0', '3.4000000000000004', '2.6500000000000004'],
                 ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.20000000000000001', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '2.1499999999999999', '0.0', '0.0', '0.35000000000000003', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.60000000000000009', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.1500000000000001', '0.0', '0.0', '0.0', '0.0', '0.0', '0.45000000000000001', '0.0', '0.0', '2.1000000000000001', '0.0', '0.0', '0.40000000000000002', '0.0', '0.0', '1.4000000000000001', '0.0'],
                 ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.8', '0.0', '0.0', '0.0', '0.15000000000000002',
                  '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '2.1000000000000001',
                  '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '3.8000000000000003',
                  '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                  '0.0', '0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '2.6500000000000004', '0.0', '0.0',
                  '0.5', '0.0', '1.5', '0.15000000000000002', '0.0', '0.0', '0.0', '1.1500000000000001',
                  '2.4500000000000002', '0.0', '0.0', '0.0', '1.25', '0.0', '3.1000000000000001', '0.0',
                  '0.15000000000000002', '0.0', '0.0', '2.5500000000000003', '0.0', '0.5', '0.0', '0.0', '0.0', '0.0',
                  '0.0', '0.0', '0.60000000000000009', '0.0', '0.0', '2.5500000000000003', '0.0', '0.0'],
                 ['0.0', '0.55000000000000004', '0.0', '0.55000000000000004', '0.0', '0.10000000000000001', '1.0', '2.4000000000000004', '4.3500000000000005', '0.35000000000000003', '0.0', '1.4000000000000001', '0.0', '0.0', '0.0', '0.0', '1.4000000000000001', '1.3500000000000001', '4.7000000000000002', '0.0', '0.0', '0.0', '0.55000000000000004', '0.0', '0.0', '0.15000000000000002', '2.0', '4.9000000000000004', '0.0', '0.0', '0.10000000000000001', '0.0', '0.10000000000000001', '0.0', '0.0', '2.9000000000000004', '0.0', '0.050000000000000003', '0.0', '0.0', '0.0', '0.0', '0.0', '4.8500000000000005', '3.5500000000000003', '0.10000000000000001', '0.25'],
                 ['0.050000000000000003', '0.30000000000000004', '0.80000000000000004', '0.0', '0.0', '0.85000000000000009', '0.80000000000000004', '1.4500000000000002', '0.0', '0.10000000000000001', '0.15000000000000002', '2.0', '0.0', '0.85000000000000009', '0.35000000000000003', '0.0', '0.55000000000000004', '1.55', '0.0', '0.20000000000000001', '0.85000000000000009', '0.050000000000000003', '0.0', '0.70000000000000007', '0.0', '0.050000000000000003', '0.75', '0.0', '0.0', '0.65000000000000002', '0.85000000000000009', '0.0', '0.0', '0.050000000000000003', '0.0', '0.0', '0.85000000000000009', '0.0', '0.85000000000000009', '0.30000000000000004', '0.0', '0.15000000000000002', '0.0', '3.8000000000000003', '0.050000000000000003', '1.8500000000000001', '0.0', '0.050000000000000003'],
                 ['0.0', '0.0', '2.8500000000000001', '3.8000000000000003', '0.0', '0.45000000000000001', '0.35000000000000003', '0.15000000000000002', '0.30000000000000004', '0.0', '0.45000000000000001', '0.0', '0.45000000000000001', '0.0', '1.0', '0.60000000000000009', '1.8500000000000001', '0.0', '0.15000000000000002', '1.8500000000000001', '0.0', '0.20000000000000001', '0.15000000000000002', '0.0', '0.10000000000000001', '0.0', '0.80000000000000004', '0.30000000000000004', '0.0', '0.0', '0.0', '0.0', '2.2000000000000002', '3.3500000000000001', '0.0', '0.0', '4.4500000000000002', '0.25', '0.85000000000000009', '0.85000000000000009', '0.0', '0.0', '0.15000000000000002', '2.7000000000000002', '0.0', '0.0', '4.8500000000000005', '0.0', '3.6500000000000004']]

    star_masses = [['7.8925920647663954', '0.08578599866347697', '0.15495912189647723', '0.029491154294131964', '0.077553734981992195', '0.14149898281968942', '2.1821394437834534', '0.79020629093881667', '0.23605581092528918', '0.08416980015069557', '0.020380724375730557', '0.27697230439350612', '0.60077564524283256', '0.30858942806204137', '0.010410710897155936', '0.050603435098855648', '0.012478275893185913', '0.29105328831684452', '0.23255862855526968', '0.11540034059712705', '0.36610757107042413', '0.045137257166174123', '0.034166004169335754', '0.10927790021675385', '0.068593530949508694', '0.14480645969656694', '0.37949444835469254', '0.69728593927047133', '0.37848311598592249', '0.3872285246216684', '2.1822702259090323', '0.092919160009982613', '0.076135186129506746', '0.17478807976424968', '0.11061132867912558', '0.23284145901479916', '0.26280415292369846', '0.053479286282756625', '0.10399244492960322', '0.054114709617868073', '0.082499541954087555', '0.10772292031180518', '0.050851346342111338', '0.013476411721204879', '0.24690174447631361', '0.20684025962992444', '0.042012049932137002', '0.098351491974785821', '0.03492010567873164', '0.082375168174714963', '0.054515641824395347', '0.013980364227905091', '0.15270405443303237', '0.077089505250823032', '0.284235938096499', '0.094636607083620788', '0.1544920763046036', '0.011964653362721019', '0.15563996938952346', '0.028019475813336883', '0.47671546552481753', '0.038769101172988284', '0.55046365689224264', '0.028285209705224881', '0.47990973117480962', '0.027558359168775644', '0.08853119532364534', '0.2421569625189802', '0.70323449627536005', '0.18020902731154631', '0.04361052036249477', '0.019260938549533813', '0.11163601852155271', '0.16904631772460063', '0.43155179300668289', '0.071578266336206384', '0.20352252865834661', '0.038523105714447052', '0.081152223727417802', '0.18733504636704321', '0.096615919500230379', '0.3255525168886601', '0.028530829968810192', '0.045865663569054436', '0.17519686400592124', '0.014264802227147978', '0.73407205525237573', '0.40849267111651222', '0.48045990479210815', '0.078327908323671994', '0.3829952419584749', '0.1320912483166945', '0.15301091166082079', '0.16263768160745962', '0.11137706899129608', '0.044107991239436928', '0.093788224658243741'],
                   ['0.10535846580581018', '0.17563445005833458', '0.068466311479039282', '0.02445389529348007', '0.040232918980178785', '0.13769139968085603', '0.1189349356904158', '0.2915004223795381', '0.068323301199609138', '0.4060288024704144', '0.10387090978195559', '0.093302942484663129', '0.022987504398385913', '0.044521282633084691', '0.015089421807336816', '0.077224871641237688', '0.059469567768664008', '0.048820279247260324', '0.43649225408746606', '0.15171468633959925', '0.41016242999927882', '0.37123593651607661', '0.030131510328727372', '0.50107264548196651', '0.16743851235963039', '0.26720672160831593', '0.48233384540286767', '0.012410032147565805', '0.08086258757999748', '0.32277111201840847', '0.016654179170214643', '0.093807303732051719', '0.082394314895196816', '0.013556591232900046', '0.34791768005895679', '0.8916249355874114', '0.092118744793631607', '0.025471869703606997', '0.020515499962752398', '0.030075870605079862', '0.055005782892337887', '0.093804021461998177', '0.065238934200255741', '0.068462766376395479', '0.47255336222553984', '0.40300063918052059', '0.097158419927097575', '0.10640557171388565', '0.85769948306920407', '0.44972563194429738', '0.066227742805161138', '0.25301585087070072', '0.10451459145029818', '0.50027394546074833', '0.18489955149519566', '3.4320118074457873', '0.014403113414064755', '0.6871540037848316', '0.12752408475861074', '0.84085847309408623', '0.45994141851627318', '1.0294631737696913', '0.11285843006292262', '0.2784267914556407', '0.13360454809417596', '0.092611535116119359', '0.080419222303985627', '0.55177286103043433', '0.016743942349973801', '0.021776201252424032', '0.018822549570510182', '0.011656429453449397', '0.11807811595487405', '0.20013118752674403', '0.27134495326103375', '1.0695387342414899', '0.067864296273020097', '0.030703961814382372', '0.32146855551665232', '0.94664971508879636', '0.37699620952780644', '0.014303113729260799', '0.11408907544712962', '0.025683016423812211', '0.0735775707306182', '0.21658582042965299', '0.043098155748294585', '1.6276417637955194', '0.021695349584543992', '0.13126189422578047', '0.064550916340976969', '0.12245091956523743', '0.84835561357849532', '1.0648523048013181', '0.39223383659469485', '0.87436029005565563', '0.014803962305050338', '0.12573225135036756', '0.2116074831456676'],
                   ['0.021372887969887681', '0.58730908362537104', '0.73877888990110852', '0.16434433549841612', '0.63367089983765845', '0.20656031721779866', '0.064666998038058254', '0.38084792096060049', '0.054857650099613131', '0.02950908740213103', '0.23730939634169979', '0.044092057621684763', '0.47309155823226695', '0.06283223982189147', '0.10960996661447364', '0.029280295815563122', '0.09882022373973369', '1.8168780464274685', '0.10466863863139718', '0.015304103582641022', '0.022670017690599127', '0.23405664310659746', '0.059449415498988904', '2.8070581702865032', '0.25601467746343237', '0.36304471698417684', '0.12260562955387715', '1.3340773868745286', '0.43846114856168122', '0.57944763728629811', '0.092770202618940076', '0.076428538387041722', '0.09794338380729832', '0.014330630002452079', '0.65497364362146959', '1.1049769547521977', '0.14587321378218721', '0.28273178668950932', '0.37243769609429533', '8.7274950533158524', '0.065375000977159467', '0.023172473751405357', '1.2879749528039834', '0.082077324775941346', '0.11294358912632994', '0.29003167115863721', '0.32078658773041818', '0.12301842339376488', '0.4135751747522074', '0.23240393240490703', '0.04506914810069905', '0.030302373161031848', '0.081108867530018808', '0.035355425947856685', '0.032914654359059006', '0.038652561508472801', '0.081430618783126546', '0.28245276472595349', '0.48876205753007074', '0.55681118478448', '0.277128430908543', '0.18080434272817883', '0.14666082679103082', '0.70551879796174144', '0.015433158699347487', '0.087831954089852546', '0.050748140951217657', '0.17747577001018475', '0.14913344885805749', '0.12212791954813849', '0.091827678606769048', '0.03444620940250024', '0.73533096343832349', '0.035839971599232634', '0.35618923324912033', '0.097910156563358108', '2.0373223630567834', '0.38178070458814262', '0.12533948912127182', '0.30198875227151273', '0.032955361211256876', '0.13947892708807894', '0.013548731630907312', '0.095315503668255849', '0.19301776427111886', '0.063848095138707442', '0.020788771756301078', '0.013935169171913134', '4.498211640509755', '0.20622841596198757', '0.045820811804869806', '0.015594595700042995', '0.027967102510848538', '0.088222341296610107', '0.044024060021399372', '0.036621486110985833'],
                   ['1.1738868320266571', '0.017890506090368356', '0.078346580509037825', '0.37333458109050477', '0.078653228928028437', '0.023559329791601819', '0.056569687920594913', '1.3658214535407045', '0.97027804518371552', '0.65033917427463128', '0.12586378837806034', '13.945874521944738', '0.068376991338460813', '0.06312836334601242', '0.098995250051522349', '0.30128081950731572', '0.038947248379774903', '0.16440804732498307', '0.07317914420898132', '0.14207802262674832', '0.21526905830045379', '0.20783290294264015', '0.031395143686878083', '0.18264551330065382', '0.1136959552170924', '0.25805469639925099', '0.036108240152649337', '0.12406410910301784', '0.46159363628454464', '1.948230132293894', '0.063228499784411488', '0.1380356804091078', '0.022309698297076522', '0.14671229569172947', '0.033925394233753521', '0.62751472915281636', '0.055347039482111983', '0.731144347982844', '0.52817085342185799', '0.39365950780958764', '0.10910547491289364', '0.12297997056231122', '1.9417300097205095', '0.033302272187667598', '0.21920004595711931', '0.094167372807628755', '0.86388719783278745'],
                   ['0.032747578937686213', '0.010524262819625832', '0.066460703441150321', '0.017817845764994369', '0.016229640805070254', '0.2963028977392978', '0.062518021886974903', '0.62093665900098871', '3.249928486821251', '0.016061546260203129', '0.05039085123443128', '0.12657737788533985', '0.49989242056798172', '0.023441569500674962', '0.075695280061334955', '0.017636095975757729', '3.7782831341779217', '0.046695509249648472', '0.040731480455497071', '0.084000048542360276', '0.076429497996356441', '0.80575084942933706', '0.075898757306575193', '0.51605326198683021', '0.28205996926451671', '0.13467893359238889', '0.052646255348797077', '0.068136169574577993', '0.03617654344500941', '0.091234600148168005', '0.09023713071724826', '0.58148544666827096', '0.10465874396699693', '0.10934800328404121', '0.029713976375554996', '0.21242671077560143', '0.089676903366058811', '0.022364469767355781', '0.071119394546358225', '0.24486767946095436', '0.18557422519895347', '0.011355326494560948', '0.062398561825529136', '0.046747710263608501', '0.03130686372865564', '0.42543404103608956', '0.20252597301424102', '0.034321333232007575'],
                   ['0.20184985209771722', '0.040944306874568917', '0.2352917829695301', '0.13398682838649303', '0.80608163305710279', '0.040187568875498035', '0.076973726659159122', '0.10961651547339443', '0.01818093981678718', '0.31081260186789', '0.041428979596553867', '0.07486416210046129', '0.094164486922922883', '0.037860543939663087', '0.044717487273878746', '0.42021875746621229', '0.15257400449146263', '0.62812811520914258', '0.066549383195718237', '0.10453468514431423', '0.020823502810227421', '1.0251483310370522', '0.17008910524328127', '0.11034705729491766', '0.19129864744465608', '0.143750399059873', '0.041533811979215934', '0.023062581339886305', '0.29724822750932289', '0.3442689844802157', '0.25712970644735583', '0.095730151273542716', '0.27067331070897083', '0.41596957864563805', '0.097287193665811234', '0.079846638632280462', '0.83973506097119444', '0.047748056400997857', '0.11099028265784527', '4.8972714671159734', '0.13895202322382169', '0.039883627551372987', '0.091829881036494032', '0.015247047472852151', '0.48746609017117765', '0.18291881343109698', '0.24833576647138933', '0.42402520947834377', '1.3578959055451456']]

    for t in max_times:
        f = []
        for d in t:
            f.append(float(d))
        max_times[max_times.index(t)] = f

    for t in star_masses:
        f = []
        for d in t:
            f.append(float(d))
        star_masses[star_masses.index(t)] = f

    pyplot.scatter(numpy.array(max_times[0]), 50. * numpy.array(mdots[0]),
                   s=100,
                   c=numpy.array(star_masses[0]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    pyplot.scatter(numpy.array(max_times[1]), 50. * numpy.array(mdots[1]),
                   s=100,
                   c=numpy.array(star_masses[1]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    pyplot.scatter(numpy.array(max_times[2]), 50. * numpy.array(mdots[2]),
                   s=100,
                   c=numpy.array(star_masses[2]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    pyplot.scatter(numpy.array(max_times[3]), numpy.array(mdots[3]),
                   s=100,
                   c=numpy.array(star_masses[3]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    pyplot.scatter(numpy.array(max_times[4]), numpy.array(mdots[4]),
                   s=100,
                   c=numpy.array(star_masses[4]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    pyplot.scatter(numpy.array(max_times[5]), numpy.array(mdots[5]),
                   s=100,
                   c=numpy.array(star_masses[5]), cmap=cm,
                   norm=matplotlib.colors.LogNorm(),
                   alpha=0.5)

    cbar = pyplot.colorbar()
    cbar.set_label(r'Stellar mass $[\mathrm{\ M}_\odot]$')
    pyplot.xticks([1, 2, 3, 4, 5])

    '''for p in open_paths50:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, 0.0)
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
                f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
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
            max_mdots.append(float(m) / 1000)
            max_times.append(times[all_mdot.index(m)])
            star_mass.append(init_masses[i])

        print max_mdots
        print max_times
        print star_mass'''

    pyplot.xlabel(r'Time [Myr]')
    pyplot.ylabel(r'$\max(\mathrm{d}\dot{\mathrm{M}}/\mathrm{dt})\ [\mathrm{\ M}_{Jup} / \mathrm{\ yr}]$')
    pyplot.ylim([0, 0.13])
    pyplot.savefig('{0}/max_mdot.png'.format(save_path))
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

        cumulative_mass(paths100, paths50, save_path)

        #mdot(paths100, paths50, save_path)

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

