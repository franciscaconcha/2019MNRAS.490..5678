import numpy
import math
import matplotlib
from matplotlib import pyplot
from amuse.lab import *
from amuse import io
import matplotlib.lines as mlines
import matplotlib.patches as patches


# Custom legend lines
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
                           color="rebeccapurple")  # Have to change color by hand for other plots
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 46),  # (x,y)
            1.2 * width,  # width
            1.4 * height,  # height
            fill='rebeccapurple',
            facecolor="rebeccapurple",
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
                           color="rebeccapurple")
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 46),  # (x,y)
            1.15 * width,  # width
            1.4 * height,  # height
            fill='rebeccapurple',
            facecolor="rebeccapurple",
            edgecolor="rebeccapurple",
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


def luminosity_fit(masses):  # For G0 calculation
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
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 30, t)
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


def mass_loss_in_time(open_paths100, open_paths30, save_path, tend, save, mass_limit=0.5):
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
        for p in open_paths30:
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

    ax.semilogx(times, photoevap_mass_loss, label="Photoevaporation", lw=3, color="#009bed")
    ax.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    facecolor="#009bed",
                    alpha=0.2)

    ax.semilogx(times, trunc_mass_loss, label="Dynamical truncations", lw=3, color="#d73027")
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
        for p in open_paths30:
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

    ax.semilogx(times, photoevap_mass_loss, label="Photoevaporation", ls="--", lw=3, color="#009bed")
    ax.fill_between(times,
                    photoevap_low,
                    photoevap_high,
                    alpha=0.2, facecolor="#009bed", edgecolor='#009bed', hatch="/")

    ax.semilogx(times, trunc_mass_loss, label="Dynamical truncations", ls="--", lw=3, color="#d73027")
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
    pyplot.xticks([0.1, 1, 5], ['0.1', '1', '5'])

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


def cdfs_with_observations_size(open_path100, open_path30, save_path, N, times, colors, labels, save, log=False):
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
        for p in open_path100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]
            #print small_stars.disk_mass.value_in(units.g) / (math.pi * small_stars.disk_radius.value_in(units.cm)**2)
            small_stars = small_stars[1. < small_stars.disk_mass.value_in(units.g) / (math.pi * small_stars.disk_radius.value_in(units.cm)**2)]

            if log: # 2. factor radii to diameter
                sizes = numpy.log10(2. * small_stars.disk_radius.value_in(units.au))
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
        xtext = 240
        ytext = 0.05
        textsize = 18
        xlimits = [0, 500]
        ylimits = [0.0, 1.0]
        ticks = [0, 250, 500]
        xlabel = '$d_{disk}$ [au]'
        ylabel = '$f < d_{disk}$'

        if t == 1.:
            # ONC data (Eisner+ 2018)
            # Data: dust radii
            lines = open('data/ONC.txt', 'r').readlines()
            onc_sizes, onc_sizes_error = [], []

            for line in (line for line in lines if not line.startswith('#')):
                data = line.split('&')[7]
                b = data.split('$')[1]
                c = b.split('\pm')
                if len(c) == 2:  # Value + error
                    onc_sizes.append(2. * 2. * float(c[0]))  # 2. factor for radius to diameter, 2. dust to gas
                    onc_sizes_error.append(2. * 2. * float(c[1]))
                else:  # Upper limit
                    d = c[0].split('<')[1][1:]
                    onc_sizes.append(2. * 2. * float(d))
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

            # 100 MSun
            axs00.plot(disk_sizes,
                       cumulative_sizes,
                       lw=3, color='black')
            axs00.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs00.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs00.set_xlim(xlimits)
            axs00.set_ylim(ylimits)
            axs00.set_xticks(ticks)

        elif t == 2.:
            # Lupus data (Ansdell et al 2018)
            # Data: gas radii
            gas_disk_sizes, gas_disk_sizes_error = [], []

            lines = open('data/Lupus_sizes.txt', 'r').readlines()
            for line in (line for line in lines if not line.startswith('#')):
                r_gas = line.split()[5]
                r_gas_error = line.split()[6]
                gas_disk_sizes.append(2. * float(r_gas))  # 2. factor radii to diameter
                gas_disk_sizes_error.append(2. * float(r_gas_error))

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
                           ls='-', lw=3,
                           color=colors[1],
                           label=labels[1])
            axs01.fill_betweenx(p,
                                    low_lupus, high_lupus,
                                    alpha='0.2', facecolor=colors[1])
            axs01.set_title('Lupus clouds')
            #axs01.legend()
            axs01.set_xlabel(xlabel)
            axs01.set_ylabel(ylabel)

            # 100 MSun
            axs01.plot(disk_sizes,
                       cumulative_sizes,
                        lw=3, color='black')
            axs01.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs01.text(490, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs01.set_xlim([0, 1050])
            axs01.set_ylim(ylimits)
            axs01.set_xticks([0, 500, 1000])

        elif t == 2.5:
            # Chamaeleon I data (Pascucci et al 2016)
            # Data: dust major axes
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
                cham_sorted_disk_sizes = numpy.sort(numpy.log10(2. * cham_sizes_au))  # 2. factor dust to gas
            else:
                cham_sorted_disk_sizes = numpy.sort(2. * cham_sizes_au)

            p = 1. * numpy.arange(len(cham_sorted_disk_sizes)) / (len(cham_sorted_disk_sizes) - 1)

            axs02.plot(cham_sorted_disk_sizes, p,
                           ls='-', lw=3,
                           color=colors[2],
                           label=labels[2])
            axs02.set_title('Chamaeleon I')
            #axs02.legend()
            axs02.set_xlabel(xlabel)
            axs02.set_ylabel(ylabel)
            axs02.set_xticks([0, 250, 500])

            # 100 MSun
            axs02.plot(disk_sizes,
                       cumulative_sizes,
                        lw=3, color='black')
            axs02.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs02.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs02.set_xlim(xlimits)
            axs02.set_ylim(ylimits)
            axs02.set_xticks(ticks)

        elif t == 4.:
            # sigma Orionis data (Mauco et al 2016)
            # Data: dust radii
            lines = open('data/sigmaOrionis_sizes.txt', 'r').readlines()
            sOrionis_sizes_au, sOrionis_sizes_low, sOrionis_sizes_high = [], [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[1]
                b = line.split()[2][1:-1]
                c, d = b.split('-')
                sOrionis_sizes_au.append(2. * 2. * float(a))  # 2. factor radii to diameter, 2. factor dust to gas
                sOrionis_sizes_low.append(2. * 2. * float(c))
                sOrionis_sizes_high.append(2. * 2. * float(d))

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
                           ls='-', lw=3,
                           color=colors[3],
                           label=labels[3])
            axs10.fill_betweenx(p,
                                    sOrionis_sorted_low, sOrionis_sorted_high,
                                    alpha='0.2', facecolor=colors[3])
            axs10.set_title('$\sigma$ Orionis')
            #axs10.legend()
            axs10.set_xlabel(xlabel)
            axs10.set_ylabel(ylabel)

            # 100 MSun
            axs10.plot(disk_sizes,
                       cumulative_sizes,
                        lw=3, color='black')
            axs10.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs10.text(490, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs10.set_xlim([0, 1050])
            axs10.set_ylim(ylimits)
            axs10.set_xticks([0, 500, 1000])

        elif t == 5.:
            # UpperSco data (Barenfeld et al 2017)
            # Data: gas radii
            lines = open('data/UpperSco_sizes.txt', 'r').readlines()
            uppersco_sizes, uppersco_errors_low, uppersco_errors_high = [], [], []

            for line in (line for line in lines if not line.startswith('#')):
                a = line.split()[7]
                b = line.split()[8]
                c = line.split()[9]
                uppersco_sizes.append(2. * float(a))  # 2. factor radii to diameter
                uppersco_errors_low.append(2. * float(b[2:-1]))
                uppersco_errors_high.append(2. * float(c[1:-1]))

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

            # 100 MSun
            axs11.plot(disk_sizes,
                       cumulative_sizes,
                        lw=3, color='black')
            axs11.fill_betweenx(cumulative_sizes,
                                 sizes_low, sizes_high,
                                 alpha='0.2', facecolor='black')
            axs11.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs11.set_xlim(xlimits)
            axs11.set_ylim(ylimits)
            axs11.set_xticks(ticks)

    pyplot.tight_layout()

    if save:
        pyplot.savefig('{0}/CDF_data_size.png'.format(save_path))

    pyplot.show()


def cdfs_with_observations_mass(open_path, save_path, N, times, colors, labels, save, log=False):
    """ Plot cumulative distributions of disk sizes (au) and masses (MJup).

    :param open_path: list of folders to use
    :param save_path: location where the figures are saved
    :param N: number of stars
    :param t: time to show in plot
    :param log: if True, plot in logscale
    """
    fig = pyplot.figure()
    axs00 = pyplot.subplot2grid((2, 6), (0, 0), colspan=2)  # 1 Myr: ONC, OMC-1, OMC-2
    axs01 = pyplot.subplot2grid((2, 6), (0, 2), colspan=2)  # 2 Myr: Lupus
    axs02 = pyplot.subplot2grid((2, 6), (0, 4), colspan=2)  # 2.5 Myr: ChamI, IC348
    axs10 = pyplot.subplot2grid((2, 6), (1, 1), colspan=2)  # 4 Myr: sigmaOrionis
    axs11 = pyplot.subplot2grid((2, 6), (1, 3), colspan=2)  # 5 Myr: UpperSco

    for t in times:
        all_sorted_disk_masses = []
        for p in open_path:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, N, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            # Take only the small stars
            small_stars = stars[stars.bright == False]
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
        xtext = 0.65
        ytext = 0.05
        textsize = 18
        xlabel = '$\log(M_{disk})$ [$M_{Jupiter}$]'
        ylabel = '$f < M_{disk}$'

        if t == 1.0:
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
                           label='ONC')
            axs00.fill_betweenx(p,
                                onc_low, onc_high,
                                alpha='0.2', facecolor=colors[0])
            axs00.set_title('ONC')
            #axs00.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)

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

            axs00.plot(sorted_omc1_masses, p,
                           ls='-', lw=3,
                           color=colors[7],
                           label='OMC-1')
            axs00.fill_betweenx(p,
                                omc1_low, omc1_high,
                                alpha='0.2', facecolor=colors[7])
            axs00.set_title('OMC-1')
            #axs21.legend()
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

            axs00.plot(sorted_omc2_masses, p,
                           ls='-', lw=3,
                           color=colors[6],
                           label='OMC-2')
            axs00.fill_betweenx(p,
                                omc2_low, omc2_high,
                                alpha='0.2', facecolor=colors[6])
            axs00.set_title('OMC-2')
            #axs20.legend()
            axs00.set_xlabel(xlabel)
            axs00.set_ylabel(ylabel)

            axs00.plot(disk_masses,
                        cumulative_masses,
                        lw=3, color='black')
            axs00.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs00.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs00.set_xlim(xlimits)
            axs00.set_ylim(ylimits)
            axs00.set_xticks(ticks)
            axs00.set_yticks(yticks)
            axs00.set_title('Orion region')
            axs00.legend(loc='upper left', fontsize=14, handlelength=0.5, framealpha=0.2)

        elif t == 2.0:
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

            axs01.plot(disk_masses,
                        cumulative_masses,
                        lw=3, color='black')
            axs01.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs01.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs01.set_xlim(xlimits)
            axs01.set_ylim(ylimits)
            axs01.set_xticks(ticks)

        elif t == 2.5:
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
                           label='ChamI')
            axs02.fill_betweenx(p,
                                cham_low, cham_high,
                                alpha='0.2', facecolor=colors[2])
            axs02.set_title('Chamaeleon I')
            #axs02.legend()
            axs02.set_xlabel(xlabel)
            axs02.set_ylabel(ylabel)

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

            axs02.plot(ic348_sorted_masses, p,
                        ls='-', lw=3,
                        color=colors[5],
                        label='IC 348')
            axs02.fill_betweenx(p,
                                ic348_low, ic348_high,
                                alpha='0.2', facecolor=colors[5])
            axs02.set_title('IC 348')
            #axs02.legend()
            axs02.set_xlabel(xlabel)
            axs02.set_ylabel(ylabel)

            axs02.plot(disk_masses,
                        cumulative_masses,
                        lw=3, color='black')
            axs02.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs02.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs02.set_xlim(xlimits)
            axs02.set_ylim(ylimits)
            axs02.set_xticks(ticks)
            axs02.set_title('ChamI \& IC 348')
            axs02.legend(loc='upper left', fontsize=14, handlelength=0.5, framealpha=0.2)

        elif t == 4.0:
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

            axs10.plot(disk_masses,
                        cumulative_masses,
                        lw=3, color='black')
            axs10.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs10.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs10.set_xlim(xlimits)
            axs10.set_ylim(ylimits)
            axs10.set_xticks(ticks)

        elif t == 5.0:
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

            axs11.plot(disk_masses,
                        cumulative_masses,
                        lw=3, color='black')
            axs11.fill_betweenx(cumulative_masses,
                                 masses_low, masses_high,
                                 alpha='0.2', facecolor='black')
            axs11.text(xtext, ytext, 't = {0} Myr'.format(t), fontsize=textsize)
            axs11.set_xlim(xlimits)
            axs11.set_ylim(ylimits)
            axs11.set_xticks(ticks)

    pyplot.tight_layout()

    if save:
         pyplot.savefig('{0}/CDF_data_mass.png'.format(save_path))

    pyplot.show()


def dist_disk_mass(open_paths100, open_paths30, save_path, t_end, save):
    fig = pyplot.figure()

    dt = 1.
    times = numpy.arange(0.0, t_end + dt, dt)

    p =open_paths100[2]

    for t in times:
        total_in_t = []
        #for p in open_paths100:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        # Take only the small stars
        small_stars = stars[stars.bright == False]
        small_stars = small_stars[small_stars.dispersed == False]

        disk_masses = small_stars.disk_mass.value_in(units.MJupiter)
        #total_in_t.append(disk_masses)

        if t == 0.:
            print small_stars.disk_mass.value_in(units.MSun)
            print small_stars.stellar_mass.value_in(units.MSun)
        #    y, bins = numpy.histogram(0.1 * stars.stellar_mass.value_in(units.MJupiter))
        #    bincenters = 0.5 * (bins[1:] + bins[:-1])
        #    pyplot.plot(bincenters, y, lw=3, label="IMF")

        #else:
        y, bins = numpy.histogram(disk_masses)
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        pyplot.plot(bincenters, y, lw=3, label="{0} Myr".format(t))


    pyplot.xlabel('Disk mass [M$_{Jup}$]')
    #pyplot.ylabel(r'$f_{> 10 \mathrm{\ M}_{\oplus}}$')
    pyplot.legend()
    #pyplot.xlim([0.0, 80.0])
    #pyplot.ylim([0.0, 100.0])
    if save:
         pyplot.savefig('{0}/mass_distribution.png'.format(save_path))
    pyplot.show()


def dist_disk_size(open_paths100, open_paths30, save_path, t_end, save):
    fig = pyplot.figure()

    dt = 1.
    times = numpy.arange(0.0, t_end + dt, dt)

    p = open_paths100[1]

    for t in times:
        total_in_t = []
        #for p in open_paths100:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        # Take only the small stars
        small_stars = stars[stars.bright == False]
        small_stars = small_stars[small_stars.dispersed == False]

        disk_masses = numpy.sort(small_stars.disk_radius.value_in(units.au))
        #total_in_t.append(disk_masses)

        y, bins = numpy.histogram(disk_masses)
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        pyplot.plot(bincenters, y, lw=3, label="{0} Myr".format(t))


    pyplot.xlabel('Disk size [au]')
    #pyplot.ylabel(r'$f_{> 10 \mathrm{\ M}_{\oplus}}$')
    pyplot.legend()
    #pyplot.xlim([0.0, 80.0])
    #pyplot.ylim([0.0, 100.0])
    if save:
        pyplot.savefig('{0}/size_distribution.png'.format(save_path))
    pyplot.show()


def disk_mass(open_paths100, open_paths30, save_path, t_end, save):
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

            disk_masses = small_stars.disk_mass.value_in(units.MEarth)

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
        for p in open_paths30:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_masses = small_stars.disk_mass.value_in(units.MEarth)

            masses = disk_masses[disk_masses > 10.]

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


def disk_size(open_paths100, open_paths30, save_path, t_end, save):
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
                init_size = float(len(sizes))

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
        for p in open_paths30:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_sizes = small_stars.disk_radius.value_in(units.au)

            sizes = disk_sizes[disk_sizes > 10.]

            if t == 0.:
                init_size = float(len(sizes))
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


def disk_fractions(open_paths100, open_paths30, t_end, save_path, save, mass_limit=0.5):
    filename = 'data/diskfractions.dat'
    f = open(filename, "r")
    lines = f.readlines()
    ages, ages_errors, disk_fraction, df_lower, df_higher = [], [], [], [], []
    relax_times = []
    src1_count = 0
    Nobs = []

    label1 = "Ribas et al (2014)"
    label2 = "Richert et al (2018)"

    print matplotlib.matplotlib_fname()

    for l in lines:
        li = l.strip()
        if not li.startswith("#"):
            x = l.split()
            ages.append(float(x[1]))
            ages_errors.append(float(x[2]))
            N = float(x[7])
            Nobs.append(N)
            relax_times.append(N / (6 * numpy.log(N)))

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

    fig = pyplot.figure(figsize=(12, 12))
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
    all_fractions = []
    all_t_relax = []
    all_density = []

    Rvir = 0.5 | units.parsec
    g = 0.11

    for p in open_paths100:
        fractions = []
        t_relax = []
        print p
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:  # Have to do this to plot in terms of initial stellar mass, not considering accreted mass
                init_mass = stars.stellar_mass
            #if open_paths100.index(p) == 2:
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum(), Rvir)
            lr = stars.LagrangianRadii(unit_converter=converter, mf=[0.5])[0][0]
            bound = stars.bound_subset(tidal_radius=lr, unit_converter=converter)
            tdyn = numpy.sqrt(Rvir**3 / (constants.G * bound.stellar_mass.sum()))
            N = len(bound)
            trh = 0.138 * (N / numpy.log(g * N)) * tdyn
            t_relax.append(1E-6 * trh.value_in(units.yr))
                #    v = [numpy.sqrt(numpy.mean(stars.vx.value_in(units.parsec / units.yr)**2)),
            #         numpy.sqrt(numpy.mean(stars.vy.value_in(units.parsec / units.yr)**2)),
            #         numpy.sqrt(numpy.mean(stars.vz.value_in(units.parsec / units.yr)**2))]
            #    vm = numpy.mean(numpy.array(v))
            #    t_relax.append(1e-6 * ((0.1 * 100 * (lr / vm)) / numpy.log(100))) # yr to Myr
            #density.append(count_stars(stars, lr))
            stars.stellar_mass = init_mass
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.:
                print stars[stars.bright == True].stellar_mass.value_in(units.MSun)

            fraction = 100. * (float(len(disked_stars)) / float(len(small_stars)))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_high = numpy.max(all_fractions, axis=0)
    disk_fractions_low = numpy.min(all_fractions, axis=0)

    pyplot.plot(times / numpy.mean(all_t_relax, axis=0),#/ (100. / (6 * numpy.log(100))),
                all_disk_fractions / 100.,
                #yerr=disk_fractions_stdev,
                color='k', lw=3,
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')
    print (100./numpy.log(100))
    #pyplot.fill_between(times / t_relax,#times / (100. / (6 * numpy.log(100))),
    #                    disk_fractions_high / 100.,
    #                    disk_fractions_low / 100.,
    #                    facecolor='black', alpha=0.2)

    all_fractions = []
    all_t_relax = []

    for p in open_paths30:
        fractions = []
        t_relax = []
        print p
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:   # Have to do this to plot in terms of initial stellar mass, not considering accreted mass
                init_mass = stars.stellar_mass
            #if open_paths30.index(p) == 2:
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum(), Rvir)
            lr = stars.LagrangianRadii(unit_converter=converter, mf=[0.5])[0][0]
            bound = stars.bound_subset(tidal_radius=lr, unit_converter=converter)
            tdyn = numpy.sqrt(Rvir**3 / (constants.G * bound.stellar_mass.sum()))
            N = len(bound)
            trh = 0.138 * (N / numpy.log(g * N)) * tdyn
            t_relax.append(1E-6 * trh.value_in(units.yr))
            #    v = [numpy.sqrt(numpy.mean(stars.vx.value_in(units.parsec / units.yr)**2)),
            #         numpy.sqrt(numpy.mean(stars.vy.value_in(units.parsec / units.yr)**2)),
            #         numpy.sqrt(numpy.mean(stars.vz.value_in(units.parsec / units.yr)**2))]
            #    vm = numpy.mean(numpy.array(v))
            #    t_relax.append(1e-6 * ((0.1 * 30 * (lr / vm)) / numpy.log(30))) # yr to Myr
            stars.stellar_mass = init_mass
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            if t == 0.:
                print stars[stars.bright == True].stellar_mass.value_in(units.MSun)

            fraction = 100. * (float(len(disked_stars)) / float(len(small_stars)))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_high = numpy.max(all_fractions, axis=0)
    disk_fractions_low = numpy.min(all_fractions, axis=0)

    pyplot.plot(times / numpy.mean(all_t_relax),#(30. / (6 * numpy.log(30))),
                all_disk_fractions / 100.,
                #yerr=disk_fractions_stdev,
                color='k',
                ls='--', lw=3,
                label=r'$\rho \sim 50 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')
    print (30. / numpy.log(30))
    #pyplot.fill_between(times / (30. / (6 * numpy.log(30))),
    #                    disk_fractions_high / 100.,
    #                    disk_fractions_low / 100.,
    #                    facecolor='black', alpha=0.2)

    pyplot.legend(framealpha=0.5)
    #pyplot.xlabel("Age [Myr]")
    pyplot.xlabel("t / t$_\mathrm{relax}$ ")
    #pyplot.xlabel("Number density")
    pyplot.ylabel("Disk fraction")
    #pyplot.xlim([0.0, 5.0])
    #pyplot.xlim([0.0, 3.2])
    #pyplot.ylim([0.0, 100.0])
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


def disk_stellar_mass(open_paths100, open_paths30, t_end, mass_limit, save_path, save):
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
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            print disked_stars[0]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            if t == 0.:
                initial_mass = numpy.sum(disked_stars.disk_mass.value_in(units.MJupiter))

            low_total_mass = numpy.sum(low_mass_stars.disk_mass.value_in(units.MJupiter))
            high_total_mass = numpy.sum(high_mass_stars.disk_mass.value_in(units.MJupiter))
            low_total_mass_fraction = low_total_mass / initial_mass
            high_total_mass_fraction = high_total_mass / initial_mass
            low_all_in_p.append(low_total_mass_fraction)
            high_all_in_p.append(high_total_mass_fraction)

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.median(low_mass_disks, axis=0)
    high_mass_disks = numpy.median(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, label="low mass M$_* \leq {0}$ M$_\odot$".format(mass_limit.value_in(units.MSun)))
    pyplot.plot(times, high_mass_disks, label=" high mass M$_* > {0}$ M$_\odot$".format(mass_limit.value_in(units.MSun)))

    low_mass_disks, high_mass_disks = [], []

    for p in open_paths30:
        low_all_in_p, high_all_in_p = [], []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 30, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            if t == 0.:
                initial_mass = numpy.sum(disked_stars.disk_mass.value_in(units.MJupiter))

            low_total_mass = numpy.sum(low_mass_stars.disk_mass.value_in(units.MJupiter))
            high_total_mass = numpy.sum(high_mass_stars.disk_mass.value_in(units.MJupiter))
            low_total_mass_fraction = low_total_mass / initial_mass
            high_total_mass_fraction = high_total_mass / initial_mass
            low_all_in_p.append(low_total_mass_fraction)
            high_all_in_p.append(high_total_mass_fraction)

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.mean(low_mass_disks, axis=0)
    high_mass_disks = numpy.mean(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, label="low mass stars", ls=':')
    pyplot.plot(times, high_mass_disks, label="high mass stars", ls=":")
    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel(r'M$_{disk} / $M$_{disk}(t = 0)$')
    pyplot.legend()
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


def main(save_path, time, N, distribution, ncells, i, all_distances, single, save):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    paths100 = ['results/final/plummer_N100_1/',
                'results/final/plummer_N100_2/',
                'results/final/plummer_N100_3/']

    paths50 = ['results/final/plummer_N50_1/',
               'results/final/plummer_N50_2/',
               'results/final/plummer_N50_3/']

    path = 'results/final/plummer_N100_1/'

    #for j in range(100):
    #    print j
    #    single_star(path, save_path + '/single_N100_1', N, j, time, all_distances)

    if single:
        single_star(path, save_path, N, i, time, all_distances)
        #tests(path, i, N, time)
    else:
        times = [1.0, 2.0, 2.5, 4.0, 5.0]
        colors = ["#638ccc", "#ca5670", "#c57c3c", "#72a555", "#ab62c0", '#0072B2', '#009E73', '#D55E00']  # colors from my prev paper
        labels = ['Trapezium cluster', 'Lupus clouds', 'Chamaeleon I', '$\sigma$ Orionis', 'Upper Scorpio', 'IC 348',
                  'ONC', "OMC-2"]
        #mass_loss_in_time(paths100, paths50, save_path, time, save, mass_limit=0.0)
        #disk_fractions(paths100, paths50, time, save_path, save, mass_limit=0.0)
        #disk_mass(paths100, paths50, save_path, time, save)
        disk_size(paths100, paths50, save_path, time, save)
        #disk_stellar_mass(paths100, paths30, time, 1.0, save_path, save)
        #disk_stellar_mass_scatter(paths, N, time, save_path, save)
        #luminosity_vs_mass(save_path, save)
        #g0_in_time(paths100, paths30, save_path, 100, 0)

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    result.add_option("-S", dest="save", type="int", default=0,
                      help="save plot? [%default]")

    # Simulation parameters
    result.add_option("-s", dest="save_path", type="string", default='/media/fran/data1/photoevap/figures',
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
    result.add_option("-c", dest="ncells", type="int", default=100,
                      help="Number of cells to be used in vader disk [%default]")
    result.add_option("-i", dest="i", type="int", default=0,
                      help="Individual star to plot [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

