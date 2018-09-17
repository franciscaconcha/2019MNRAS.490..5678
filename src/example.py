from amuse.lab import *
import numpy


def column_density(r):
    rd = 0.1 | units.AU
    rc = 10 | units.AU
    Md = 1 | units.MSun

    if r < rd:
        return 1E-12

    Sigma_0 = Md / (2 * numpy.pi * rc**2 * (1 - numpy.exp(-rd/rc)))
    Sigma = Sigma_0 * (r/rc) * numpy.exp(-r/rc)
    return Sigma


def main():
    disk = vader()
    disk.initialize_code()
    disk.initialize_keplerian_grid(
        128,  # Number of cells
        True,  # Linear?
        0.1 | units.AU,  # Rmin
        10 | units.AU,  # Rmax
        1 | units.MSun  # Mass
    )

    disk.set_verbosity(1)

    disk.set_grid_column_density(column_density(10 | units.AU))

    disk.evolve_model(10 | units.Myr)
    disk.stop()


if __name__ == '__main__':
    main()
