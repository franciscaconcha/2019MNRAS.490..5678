from amuse.lab import *
import numpy


def column_density(r):
    rd = 0.1 | units.AU
    rc = 10 | units.AU
    Md = 1 | units.MSun

    #if r < rd:
    #    return 1E-12

    Sigma_0 = Md / (2 * numpy.pi * rc**2 * (1 - numpy.exp(-rd/rc)))
    Sigma = Sigma_0 * (r/rc) * numpy.exp(-r/rc)
    return Sigma


def main():
    disk = vader(redirection='none')
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        128,  # Number of cells
        True,  # Linear?
        0.1 | units.AU,  # Rmin
        10 | units.AU,  # Rmax
        1 | units.MSun  # Mass
    )

    disk.parameters.verbosity = 1

    sigma = column_density(disk.grid.r)
    disk.grid.column_density = sigma

    # The pressure follows the ideal gas law with a mean molecular weight of 2.33 hydrogen masses.
    # Since gamma ~ 1, this should not influence the simulation
    mH = 1.008 * constants.u
    T = 100. | units.K
    disk.grid.pressure = sigma * constants.kB * T / (2.33 * mH)

    disk.evolve_model(0.04 | units.Myr)
    disk.stop()


if __name__ == '__main__':
    main()
