import numpy
import matplotlib
from matplotlib import pyplot
import matplotlib.lines as mlines
import matplotlib.patches as patches

from amuse.lab import *
from amuse import io


# START custom legend stuff
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

class t000Object(object):
    pass

class t005Object(object):
    pass

class t250Object(object):
    pass

class t500Object(object):
    pass

class LowMassObject(object):
    pass

class HighMassObject(object):
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
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.3 * height, 0.3 * height],
                           lw=3,
                           color='black')  # Have to change color by hand for different plots
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 46),  # (x,y)
            1.15 * width,  # width
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
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.3 * height, 0.3 * height],
                           lw=3, ls="--",
                           color='black')
        l2 = patches.Rectangle(
            (x0 - 1, y0 + width - 46),  # (x,y)
            0 * 1.15 * width,  # width
            0 * 1.4 * height,  # height
            fill='black',
            facecolor='black',
            edgecolor='black',
            alpha=0.2,
            hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class t000ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color='#1373b2')
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.1 * height, 0.1 * height],
                           lw=3, ls="--",
                           color='#1373b2')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class t005ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color='#42a6cc')
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.1 * height, 0.1 * height],
                           lw=3, ls="--",
                           color='#42a6cc')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class t250ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color='#7cccc4')
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.1 * height, 0.1 * height],
                           lw=3, ls="--",
                           color='#7cccc4')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class t500ObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 3],
                           [0.5 * height, 0.5 * height],
                           lw=3,
                           color='#b4e2ba')
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.1 * height, 0.1 * height],
                           lw=3, ls="--",
                           color='#b4e2ba')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class LowMassObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color="orange")
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color="orange")
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class HighMassObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color="mediumpurple")
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color="mediumpurple")
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]

# END custom legend stuff


def get_disk_radius(disk,
                    density_limit=1E-10):
    """ Calculate the radius of a disk in a vader grid.

    :param disk: vader disk
    :param density_limit: density limit to designate disk border
    :return: disk radius in units.au
    """
    prev_r = disk.grid[0].r

    for i in range(len(disk.grid.r)):
        cell_density = disk.grid[i].column_density.value_in(units.g / units.cm ** 2)
        if cell_density < density_limit:
            return prev_r.value_in(units.au) | units.au
        prev_r = disk.grid[i].r

    return prev_r.value_in(units.au) | units.au


def column_density(grid,
                   rc,
                   mass,
                   lower_density=1E-12 | units.g / units.cm**2):
    """ Disk column density definition as in Eqs. 1, 2, and 3 of the paper.
        (Lynden-Bell & Pringle, 1974: Anderson et al. 2013)

    :param grid: disk grid
    :param rc: characteristic disk radius
    :param mass: disk mass
    :param lower_density: density limit for defining disk edge
    :return: disk column density in g / cm**2
    """
    r = grid.value_in(units.au) | units.au
    rd = rc  # Anderson et al. 2013
    Md = mass

    Sigma_0 = Md / (2 * numpy.pi * rc ** 2 * (1 - numpy.exp(-rd / rc)))
    Sigma = Sigma_0 * (rc / r) * numpy.exp(-r / rc) * (r <= rc) + lower_density
    return Sigma


def initialize_vader_code(disk_radius,
                          disk_mass,
                          alpha,
                          r_min=0.05 | units.au,
                          r_max=2000 | units.au,
                          n_cells=100,
                          linear=True):
    """ Initialize vader code for given parameters.

    :param disk_radius: disk radius. Must have units.au
    :param disk_mass: disk mass. Must have units.MSun
    :param alpha: turbulence parameter for viscosity, adimensional
    :param r_min: minimum radius of vader grid. Must have units.au
    :param r_max: maximum radius of vader grid. Must have units.au
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
    disk.parameters.inner_pressure_boundary_torque = 0.0 | units.g * units.cm ** 2 / units.s ** 2
    disk.parameters.alpha = alpha
    disk.parameters.maximum_tolerated_change = 1E99

    return disk


def radius_plot():
    """ Figure 1: isolated disk radius evolution.
    """

    disk = initialize_vader_code(100 | units.au, 0.1 | units.MSun, 1E-4, r_max=5000 | units.au, n_cells=100, linear=False)

    pyplot.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), color='k', lw=2, label='t = 0.0 Myr')
    pyplot.axvline(get_disk_radius(disk, density_limit=1E-8).value_in(units.au), color='red', lw=2)

    disk.evolve_model(0.1 | units.Myr)

    pyplot.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), color='k', lw=2, ls='--', label='t = 0.1 Myr')
    pyplot.axvline(get_disk_radius(disk, density_limit=1E-11).value_in(units.au), color='red', lw=2, ls='--')

    disk.evolve_model(1 | units.Myr)

    pyplot.loglog(disk.grid.r.value_in(units.au), disk.grid.column_density.value_in(units.g / units.cm**2), color='k', lw=2, ls=':', label='t = 1.0 Myr')
    pyplot.axvline(get_disk_radius(disk, density_limit=1E-11).value_in(units.au), color='red', lw=2, ls=':')

    pyplot.xlim([0, 4000])
    pyplot.xlabel('Disk radius [au]')
    pyplot.ylabel('Surface density [g / cm$^2$]')
    pyplot.legend(loc='lower left')
    pyplot.savefig('radii_density.png')
    pyplot.show()


def luminosity_fit(masses):
    """ Return stellar luminosity (in LSun) for corresponding mass, as calculated with Martijn's fit

    :param masses: list of stellar masses in MSun
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
    """ Parravano et al 2003 (ApJ 584 797) luminosity fit for FUV luminosity vs stellar mass plot (Figure 2)

    :param masses: list of stellar masses in MSun
    :return: stellar luminosity in LSun
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
    """ Figure 2 of paper: FUV luminosity vs stellar mass.

    :param save_path: path where to save the figure
    :param save: True if the figure should be saved
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
    pyplot.xticks([1, 10, 100])
    if save:
        pyplot.savefig('{0}/luminosity_fit.png'.format(save_path))
    pyplot.show()


def mass_loss_in_time(open_paths100, open_paths50, save_path, tend, mass_limit=0.0, save=False):
    """ Figure 5: Mean mass loss in time due to external photoevaporation (blue) and
    dynamical truncations (red). The solid and dashed lines correspond to the rho ~ 100 MSun/pc
    and rho ~ 50 MSun/pc respectively.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save the figure
    :param tend: end time of the simulation
    :param save: if True, figure is saved
    :param mass_limit: mass limit for plot
    """
    times = numpy.arange(0.0, tend + 0.05, 0.05)

    fig = pyplot.figure()
    ax = pyplot.gca()
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'Mean mass loss [M$_{Jup}$]')

    photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []

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

        mean_photoevap = numpy.mean(photoevap_in_t)
        std_photoevap =  numpy.std(photoevap_in_t)
        photoevap_mass_loss.append(mean_photoevap)
        photoevap_low.append(mean_photoevap - std_photoevap)
        photoevap_high.append(mean_photoevap + std_photoevap)

        mean_trunc = numpy.mean(trunc_in_t)
        std_trunc =  numpy.std(trunc_in_t)
        trunc_mass_loss.append(mean_trunc)
        trunc_low.append(mean_trunc - std_trunc)
        trunc_high.append(mean_trunc + std_trunc)

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

    photoevap_mass_loss, trunc_mass_loss, photoevap_low, photoevap_high, trunc_low, trunc_high = [], [], [], [], [], []

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

        mean_photoevap = numpy.mean(photoevap_in_t)
        std_photoevap =  numpy.std(photoevap_in_t)
        photoevap_mass_loss.append(mean_photoevap)
        photoevap_low.append(mean_photoevap - std_photoevap)
        photoevap_high.append(mean_photoevap + std_photoevap)

        mean_trunc = numpy.mean(trunc_in_t)
        std_trunc =  numpy.std(trunc_in_t)
        trunc_mass_loss.append(mean_trunc)
        trunc_low.append(mean_trunc - std_trunc)
        trunc_high.append(mean_trunc + std_trunc)

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

    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([0, 25])
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

    if save:
        pyplot.savefig('{0}/mass_loss.png'.format(save_path))
    pyplot.show()


def disk_mass(open_paths100, open_paths50, save_path, t_end, save):
    """ Figure 10: fraction of disks with masses Mdisk > 10 MEarth, in time, for the two different regions.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save figure
    :param t_end: end time of simulations
    :param save: if True, save figure
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

            disk_masses = 1E-2 * small_stars.disk_mass.value_in(units.MEarth)  # 1E-2 factor: total mass to "dust" mass

            masses = disk_masses[disk_masses > 10.]

            if t == 0.:
                init_mass = float(len(masses))

            total_in_t.append(len(masses) / init_mass)

        mean = numpy.mean(total_in_t)
        std = numpy.std(total_in_t)
        total_disks.append(mean)
        total_disks_low.append(mean - std)
        total_disks_high.append(mean + std)

    pyplot.plot(times,
                total_disks,
                lw=3,
                color='darkolivegreen',
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')
    pyplot.fill_between(times,
                        total_disks_low,
                        total_disks_high,
                        alpha=0.2, facecolor='darkolivegreen')

    # 50 MSun
    total_disks, total_disks_low, total_disks_high = [], [], []
    init_mass = 0.

    for t in times:
        total_in_t = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            # Take only the small stars
            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.dispersed == False]

            disk_masses = small_stars.disk_mass.value_in(units.MEarth)

            masses = 1E-2 * disk_masses[disk_masses > 10.]  # 1E-2 factor: total mass to "dust" mass

            if t == 0.:
                init_mass = float(len(masses))

            total_in_t.append(len(masses) / init_mass)

        mean = numpy.mean(total_in_t)
        std = numpy.std(total_in_t)
        total_disks.append(mean)
        total_disks_low.append(mean - std)
        total_disks_high.append(mean + std)

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
    pyplot.ylabel(r'$f_{\mathrm{M}_\mathrm{disk} > 10 \mathrm{\ M}_{\oplus}}$', fontsize=30)
    pyplot.legend([M100shadedObject, M30shadedObject],
                  [r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                   r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
                handler_map={M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
                loc='best',
                fontsize=22, framealpha=1.)
    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0.0, 1.0])
    if save:
        pyplot.savefig('{0}/mass_fraction_line.png'.format(save_path))
    pyplot.show()


def disk_size(open_paths100, open_paths50, save_path, t_end, save):
    """ Figure 11: fraction of disks with radii Rdisk > 50 au, in time, for the two different regions.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save figure
    :param t_end: end time of simulations
    :param save: if True, save figure
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

        mean = numpy.mean(total_in_t)
        std = numpy.std(total_in_t)
        total_disks.append(mean)
        total_disks_low.append(mean - std)
        total_disks_high.append(mean + std)

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

            if t == 0.:
                init_size = float(len(small_stars))

            total_in_t.append(len(sizes) / init_size)

        mean = numpy.mean(total_in_t)
        std = numpy.std(total_in_t)
        total_disks.append(mean)
        total_disks_low.append(mean - std)
        total_disks_high.append(mean + std)

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
    pyplot.ylabel(r'$f_{\mathrm{R}_\mathrm{disk} > 50 \mathrm{\ au}}$', fontsize=30)

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


def disk_fractions(open_paths100, open_paths50, t_end, save_path, save, mass_limit=0.0):
    """ Figure 9: fraction of stars with disks as a function of time, compared to observed disk fractions.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save figure
    :param t_end: end time of simulations
    :param save: if True, save figure
    :param mass_limit: lower stellar mass limit for plot
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


    # Building the comoving binned mean line for the observational data
    # Each bin contains 10 observation points
    from scipy import stats

    tt = list(numpy.array(ages) / numpy.array(relax_times))
    sorted_tt = numpy.sort(tt)
    obs_disk_fractions = numpy.array(disk_fraction) / 100.
    sorted_disk_fractions = [x for _, x in sorted(zip(tt, obs_disk_fractions))]

    means = []
    for i in range(len(sorted_tt)):
        if i + 10 <= len(sorted_tt):
            means.append(numpy.mean(sorted_disk_fractions[i:i+10]))
            print "calculating mean between {0}, {1}".format(i, i + 10)
            print sorted_disk_fractions[i:i+10]
            print numpy.mean(sorted_disk_fractions[i:i+10]
        else:
            means.append(numpy.mean(sorted_disk_fractions[i:]))

    """my_bin_edges.sort()
    my_bin_edges[0] = 0.0  # To start from the edge of the plot, not the edge of the data
    my_bin_edges = my_bin_edges + [3.0]  # To reach the end of the plot, not just the end of the data
    print my_bin_edges

    bin_means, bin_edges, binnumber = stats.binned_statistic(numpy.array(ages) / numpy.array(relax_times), # x
                                                             numpy.array(disk_fraction) / 100.,            # values
                                                             statistic='mean',
                                                             bins=my_bin_edges)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    bins_color = '#fc9f5b'
    pyplot.hlines(bin_means, bin_edges[:-1], bin_edges[1:], color=bins_color, lw=2, linestyle="--", alpha=0.5)
    """
    pyplot.plot(sorted_tt, means, lw=3, color='#fc9f5b', label='Binned mean of observations', alpha=0.7)


    # Plotting the simulations
    """times = numpy.arange(0.0, t_end + 0.05, 0.05)

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

            fraction = float(len(disked_stars)) / float(len(small_stars))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_stdev = numpy.std(all_fractions, axis=0)
    disk_fractions_high = all_disk_fractions + disk_fractions_stdev
    disk_fractions_low = all_disk_fractions - disk_fractions_stdev

    pyplot.plot(times / numpy.mean(all_t_relax, axis=0),
                all_disk_fractions,
                color='k', lw=3,
                label=r'$\rho \sim 100 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')

    pyplot.fill_between(times / numpy.mean(all_t_relax, axis=0),
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

            fraction = float(len(disked_stars)) / float(len(small_stars))
            fractions.append(fraction)

        all_fractions.append(fractions)
        all_t_relax.append(t_relax)

    all_disk_fractions = numpy.mean(all_fractions, axis=0)
    disk_fractions_high = all_disk_fractions + numpy.std(all_fractions, axis=0)
    disk_fractions_low = all_disk_fractions - numpy.std(all_fractions, axis=0)

    pyplot.plot(times / numpy.mean(all_t_relax, axis=0),
                all_disk_fractions,
                color='k',
                ls='--', lw=3,
                label=r'$\rho \sim 50 \mathrm{ \ M}_{\odot} \mathrm{ \ pc}^{-3}$')

    pyplot.fill_between(times / numpy.mean(all_t_relax, axis=0),
                        disk_fractions_high,
                        disk_fractions_low,
                        facecolor='black', alpha=0.2)"""

    # Putting the "binned mean" label at the bottom of the legend...
    ax = pyplot.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    labels = list(labels)
    handles = list(handles)
    templabel = labels[-1]
    temphandle = handles[-1]
    labels[-1] = labels[2]
    handles[-1] = handles[2]
    labels[2] = templabel
    handles[2] = temphandle
    ax.legend(handles, labels, fontsize=16, framealpha=0.5)

    pyplot.xlabel("t / t$_\mathrm{relax}$ ")
    pyplot.ylabel("Disk fraction")
    pyplot.xlim([0.0, 3.0])
    pyplot.ylim([0.0, 1.0])

    if save:
        pyplot.savefig('{0}/disk_fraction.png'.format(save_path))

    pyplot.show()


def disk_stellar_mass(open_paths100, open_paths50, t_end, save_path, mass_limit=0.0, save=False):
    fig = pyplot.figure()
    ax = pyplot.gca()
    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    p = open_paths100[0]
    mass_limit = mass_limit | units.MSun
    initial_mass = 0.0

    low_mass_disks, high_mass_disks = [], []

    colors = ['orange', 'mediumpurple']

    for p in open_paths100:
        low_all_in_p, high_all_in_p = [], []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:
                initial_mass = stars.stellar_mass
                initial_small = stars[stars.bright == False].stellar_mass
            stars.stellar_mass = initial_mass
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            low_all_in_p.append(float(len(low_mass_stars)) / len(initial_small[initial_small < mass_limit]))
            high_all_in_p.append(float(len(high_mass_stars)) / len(initial_small[initial_small >= mass_limit]))

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.mean(low_mass_disks, axis=0)
    low_mass_disks_std = numpy.std(low_mass_disks, axis=0)
    high_mass_disks = numpy.mean(high_mass_disks, axis=0)
    high_mass_disks_std = numpy.std(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, lw=3, color=colors[0])
    pyplot.plot(times, high_mass_disks, lw=3, color=colors[1])

    pyplot.fill_between(times,
                        low_mass_disks + low_mass_disks_std,
                        low_mass_disks - low_mass_disks_std,
                        facecolor=colors[0], edgecolor=colors[0], alpha=0.2)

    pyplot.fill_between(times,
                        high_mass_disks + high_mass_disks_std,
                        high_mass_disks - high_mass_disks_std,
                        facecolor=colors[1], edgecolor=colors[1], alpha=0.2)

    low_mass_disks, high_mass_disks = [], []

    for p in open_paths50:
        low_all_in_p, high_all_in_p = [], []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            if t == 0.:
                initial_mass = stars.stellar_mass
                initial_small = stars[stars.bright == False].stellar_mass

            stars.stellar_mass = initial_mass
            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]

            high_mass_stars = disked_stars[disked_stars.stellar_mass > mass_limit]
            low_mass_stars = disked_stars[disked_stars.stellar_mass <= mass_limit]

            low_all_in_p.append(float(len(low_mass_stars)) / len(initial_small[initial_small < mass_limit]))
            high_all_in_p.append(float(len(high_mass_stars)) / len(initial_small[initial_small >= mass_limit]))

        low_mass_disks.append(low_all_in_p)
        high_mass_disks.append(high_all_in_p)

    low_mass_disks = numpy.mean(low_mass_disks, axis=0)
    low_mass_disks_std = numpy.std(low_mass_disks, axis=0)
    high_mass_disks = numpy.mean(high_mass_disks, axis=0)
    high_mass_disks_std = numpy.std(high_mass_disks, axis=0)

    pyplot.plot(times, low_mass_disks, ls='--', lw=3, color=colors[0])
    pyplot.plot(times, high_mass_disks, ls="--", lw=3, color=colors[1])

    #pyplot.fill_between(times,
    #                    low_mass_disks + low_mass_disks_std,
    #                    low_mass_disks - low_mass_disks_std,
    #                    facecolor=colors[0], edgecolor=colors[0], alpha=0.2, hatch='/')

    #pyplot.fill_between(times,
    #                    high_mass_disks + high_mass_disks_std,
    #                    high_mass_disks - high_mass_disks_std,
    #                    facecolor=colors[1], edgecolor=colors[1], alpha=0.2, hatch='/')


    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel('Disk fraction')
    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0.0, 1.0])

    ax.legend([LowMassObject(), HighMassObject(), M100shadedObject, M30shadedObject],
               [r"$\mathrm{M}_* < \mathrm{\ }$" + "{0}".format(mass_limit.value_in(units.MSun)) +  r"$\mathrm{\ M}_{\odot}$",
                r"$\mathrm{M}_* \geq \mathrm{\ }$" + "{0}".format(mass_limit.value_in(units.MSun)) +  r"$\mathrm{\ M}_{\odot}$",
                r'$\rho \sim 100 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$',
                r'$\rho \sim 50 \mathrm{\ M}_\odot \mathrm{\ pc}^{-3}$'],
               handler_map={LowMassObject: LowMassObjectHandler(),
                            HighMassObject: HighMassObjectHandler(),
                            M100shadedObject: M100shadedObjectHandler(),
                            M30shadedObject: M30shadedObjectHandler()},
               fontsize=20)#, framealpha=)

    if save:
        pyplot.savefig('{0}/stellar_mass.png'.format(save_path))

    pyplot.show()


def deltat(open_paths100, open_paths50, save_path, mass_limit=0.0, save=False):

    t_end = 5.0   # | units.Myr
    times = numpy.arange(0.0, t_end + 0.05, 0.05)

    for p in open_paths100:
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)))  # Number of disks that lost more than 5MJup in t

        cumulative = numpy.array(all_mdot).cumsum()

        pyplot.plot(times, cumulative, lw=3, c='navy')

    for p in open_paths50:
        all_mdot = []
        for t in times:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            small_stars = small_stars[small_stars.stellar_mass.value_in(units.MSun) >= mass_limit]
            disked_stars = small_stars[small_stars.dispersed == False]

            mdot = disked_stars.cumulative_photoevap_mass_loss.value_in(units.MJupiter)# - init_mdot

            # Disks that lost more than 5 MJup in dt
            mdot_5MJup = mdot[mdot >= 5.0]
            all_mdot.append(float(len(mdot_5MJup)))  # Number of disks that lost more than 5MJup in t

        cumulative = numpy.array(all_mdot).cumsum()

        pyplot.plot(times, cumulative, lw=3, c='navy', ls="--")

    pyplot.xlabel(r'Time [Myr]', fontsize=28)
    pyplot.ylabel(r'$\delta \mathrm{M} > 5 \mathrm{\ M_{Jup}}$', fontsize=30)
    pyplot.xlim([0.0, 5.0])
    pyplot.ylim([0, 1400])
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
    """ Figure 8: cumulative distribution of disk mass at different moments in time.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save figure
    """
    times = [0.0, 0.05, 2.5, 5.0]
    colors = ['#1373b2', '#42a6cc', '#7cccc4', '#b4e2ba', '#daf0d4']

    i = 0  # For colors

    for t in times:
        all_masses = []
        for p in open_paths100:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]
            disks_mass = disked_stars.disk_mass.value_in(units.MJupiter)

            sorted_disks_mass = numpy.sort(disks_mass)
            all_masses.append(sorted_disks_mass)

        try:
            disk_masses = numpy.sum(all_masses, axis=0)
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

        cumulative = 1. * numpy.arange(len(disk_masses)) / (len(stars) - 1)
        pyplot.plot(disk_masses, cumulative,
                    c=colors[i], lw=3)

        i += 1

    i = 0

    for t in times:
        all_masses = []
        for p in open_paths50:
            f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, t)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)

            small_stars = stars[stars.bright == False]
            disked_stars = small_stars[small_stars.dispersed == False]
            disks_mass = disked_stars.disk_mass.value_in(units.MJupiter)

            sorted_disks_mass = numpy.sort(disks_mass)
            all_masses.append(sorted_disks_mass)

        try:
            disk_masses = numpy.mean(all_masses, axis=0)
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

        cumulative = 1. * numpy.arange(len(disk_masses)) / (len(stars) - 1)
        pyplot.plot(disk_masses, cumulative,
                    c=colors[i], lw=3, ls="--")

        i += 1

    pyplot.xlim([0.0, 150])
    pyplot.ylim([0.0, 1.0])
    pyplot.xlabel(r'$\mathrm{M}_{\mathrm{disk\ }} [\mathrm{M_{Jup}}]$')
    pyplot.ylabel(r'$f_{\mathrm{M}_{\mathrm{disk}}}$', fontsize=30)

    # Custom legend
    ax = pyplot.gca()
    ax.legend([t000Object(), t005Object(), t250Object(), t500Object()],
              ['0.00 Myr', '0.05 Myr', '2.50 Myr', '5.00 Myr'],
              handler_map={t000Object: t000ObjectHandler(),
                           t005Object: t005ObjectHandler(),
                           t250Object: t250ObjectHandler(),
                           t500Object: t500ObjectHandler()},
              loc='best', fontsize=20, framealpha=1.)

    pyplot.savefig('{0}/cdf_disk_mass.png'.format(save_path))
    pyplot.show()


def mdot(open_paths100, open_paths50, save_path):
    """ Figure 7: moment of maximum mass loss for each disk.

    :param open_paths100: path to the rho ~ 100 MSun/pc results
    :param open_paths50: path to the rho ~ 50 MSun/pc results
    :param save_path: path to save figure
    """

    tend = 5.0
    times = numpy.arange(0.0, tend + 0.05, 0.05)
    fig = pyplot.figure()
    cm = pyplot.cm.get_cmap('RdYlBu')

    for p in open_paths100:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 100, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        init_masses = stars.stellar_mass.value_in(units.MSun)
        small_stars = stars[stars.bright == False]
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
                small_stars = stars[stars.bright == False]

                all_mdot.append(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot)
                prev_mdot = small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            m = max(all_mdot)
            max_mdots.append(float(m) / 1000)
            max_times.append(times[all_mdot.index(m)])
            star_mass.append(init_masses[i])

        pyplot.scatter(max_times, max_mdots,
                       s=100,
                       c=star_mass, cmap=cm,
                       norm=matplotlib.colors.LogNorm(),
                       alpha=0.5)

    for p in open_paths50:
        f = '{0}/N{1}_t{2}.hdf5'.format(p, 50, 0.0)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
        init_masses = stars.stellar_mass.value_in(units.MSun)
        small_stars = stars[stars.bright == False]
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
                small_stars = stars[stars.bright == False]

                all_mdot.append(small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter) - prev_mdot)
                prev_mdot = small_stars[i].cumulative_photoevap_mass_loss.value_in(units.MJupiter)

            m = max(all_mdot)
            max_mdots.append(float(m) / 1000)
            max_times.append(times[all_mdot.index(m)])
            star_mass.append(init_masses[i])

        pyplot.scatter(max_times, max_mdots,
                       s=100,
                       c=star_mass, cmap=cm,
                       norm=matplotlib.colors.LogNorm(),
                       alpha=0.5)

    cbar = pyplot.colorbar()
    cbar.set_label(r'Stellar mass $[\mathrm{\ M}_\odot]$')
    pyplot.xticks([0, 1, 2, 3, 4, 5])

    pyplot.xlabel(r'Time [Myr]')
    pyplot.ylabel(r'$\max(\mathrm{d}\dot{\mathrm{M}}/\mathrm{dt})\ [\mathrm{\ M}_\mathrm{Jup} / \mathrm{\ yr}]$')
    pyplot.ylim([0, 0.13])
    pyplot.xlim([-0.05, 5])
    pyplot.savefig('{0}/max_mdot.png'.format(save_path))
    pyplot.show()


def main(save_path, time, save):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    paths100 = ['results/final/plummer_N100_1/',
                'results/final/plummer_N100_2/',
                'results/final/plummer_N100_3/']

    paths50 = ['results/final/plummer_N50_1/',
               'results/final/plummer_N50_2/',
               'results/final/plummer_N50_3/']

    #radius_plot()
    #disk_stellar_mass(paths100, paths50, time, save_path, mass_limit=0.5, save=True)
    #mass_loss_in_time(paths100, paths50, save_path, time, mass_limit=0.0, save=False)

    disk_fractions(paths100, paths50, time, save_path, save=True)#, mass_limit=0.0)

    #deltat(paths100, paths50, save_path, save=True)

    #cumulative_mass(paths100, paths50, save_path)

    #mdot(paths100, paths50, save_path)

    #disk_mass(paths100, paths50, save_path, time, save=True)
    #disk_size(paths100, paths50, save_path, time, save=True)

    #disk_stellar_mass(paths100, paths50, time, save_path, mass_limit=0.5, save=True)
    #luminosity_vs_mass(save_path, save)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    result.add_option("-S", dest="save", type="int", default=0,
                      help="save plot? [%default]")
    result.add_option("-s", dest="save_path", type="string", default='/media/fran/data1/photevap-paper/figures',
                      help="path to save the results [%default]")
    result.add_option("-t", dest="time", type="float", default='5.0',
                      help="end time to use for plots [%default]")

    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)

