# 2019MNRAS.490..5678C External photoevaporation of circumstellar disks constrains the timescale for planet formation
#### Francisca Concha-Ramírez, Martijn J. C. Wilhelm, Simon Portegies Zwart, Thomas J. Haworth
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3537676.svg)](https://doi.org/10.5281/zenodo.3537676) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![python](https://img.shields.io/badge/python-2.7-yellow.svg)

(Code to run the simulations and generate figures of the paper: [2019MNRAS.490..5678C External photoevaporation of circumstellar disks constrains the timescale for planet formation](https://doi.org/10.1093/mnras/stz2973).)

For questions please contact Francisca Concha-Ramírez, fconcha at strw.leidenuniv.nl

## Getting Started

### Prerequisites
* Python 2.7. Should work fine with Python 3 but it has not been tested.
* AMUSE: https://github.com/amusecode/amuse
* vader: https://bitbucket.org/krumholz/vader/src
* scipy

### Running the simulations

You can run an individual simulation by using the AMUSE script directly:

```
amuse.sh vader_cluster.py
```

The script has extensive options which can be passed through the command line. For a list of these options run:

```
amuse.sh vader_cluster.py --help
```

### Creating the plots

All figures of the paper (except Figures 3 and 4, which are flow charts) were created using the script ``plots.py``:

```
amuse.sh plots.py
```
A list of options is available for this script, including the path to the files that you want to use for the plots. To see the list of options add ```--help``` or ```-h``` to the line above.

The tests and figures of the appendix can be created using the script ``tests.py``:

```
amuse.sh tests.py
```

## License

This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details
