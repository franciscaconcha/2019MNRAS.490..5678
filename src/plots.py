from matplotlib import rc
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.cm as cm
from amuse.lab import *
import numpy



rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
mpl.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
mpl.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels
from matplotlib.lines import Line2D

# Had to do this for now as a workaround, will try to get rid of it soon
x = numpy.arange(9)
ys = [i+x+(i*x)**2 for i in range(9)]

colors = cm.rainbow(numpy.linspace(0, 1, len(ys)))

sizes_p = [97.5984449702, 165.785253502, 78.9590107023, 24.613597775, 1241.47304932, 3.65430086592, 24.613597775, 51.6796184557, 1241.47304932] | units.au
masses_p = [1.349e-02, 3.163e-03, 1.555e-02, 8.380e-03, 4.659e-02, 5.685e-04, 8.441e-03, 3.396e-03, 4.297e-02] | units.MSun
m_mjup_p = masses_p.value_in(units.MJupiter)
stellar_masses_p = [0.1954458598862483, 0.04073274058924652, 0.2303392261546068, 0.15084778305151647, 0.5024875051141094, 0.014344773454271226, 0.1525009744438118, 0.05235358622040297, 0.4479253678075745] | units.MSun

sizes_np = [1241.47304932, 281.61053475, 1241.47304932, 1241.47304932, 1241.47304932, 1241.47304932, 1241.47304932, 1241.47304932, 1241.47304932] | units.au
masses_np = [1.866e-02, 3.372e-03, 2.229e-02, 1.460e-02, 4.659e-02, 1.394e-03, 1.471e-02, 5.065e-03, 4.297e-02] | units.MSun
m_mjup_np = masses_np.value_in(units.MJupiter)
#stellar_masses_np = [0.1954458598862483, 0.04073274058924652, 0.2303392261546068, 0.15084778305151647, 0.5024875051141094, 0.014344773454271226, 0.1525009744438118, 0.05235358622040297, 0.4479253678075745] | units.MSun


fig = pyplot.figure(figsize=(12, 8))
ax = pyplot.gca()

ax.scatter(sizes_p.value_in(units.au), m_mjup_p, s=1500 * stellar_masses_p.value_in(units.MSun),
           c=colors, linestyle=":", lw=2, alpha=0.5, label="photoevap")
ax.scatter(sizes_np.value_in(units.au), m_mjup_np, s=1500 * stellar_masses_p.value_in(units.MSun),
           c=colors, lw=2, alpha=0.5, label="NO photoevap")

ax.legend(loc='upper left', fontsize=20)
leg = ax.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')

i = 1
n = '%.2f' % (stellar_masses_p.value_in(units.MSun)[i])
s = str(n) + r'$M_{\odot}$'
#ax.annotate(s, (sizes_p.value_in(units.au)[i], m_mjup_p[i]), xytext=(sizes_p.value_in(units.au)[i]*1.02, m_mjup_p[i]),
#                    fontsize=14)

i = 2
n = '%.2f' % (stellar_masses_p.value_in(units.MSun)[i])
s = str(n) + r'$M_{\odot}$'
#ax.annotate(s, (sizes_p.value_in(units.au)[i], m_mjup_p[i]), xytext=(sizes_p.value_in(units.au)[i]*1.02, m_mjup_p[i]),
#                    fontsize=14)

i = len(stellar_masses_p) - 1
n = '%.2f' % (stellar_masses_p.value_in(units.MSun)[i])
s = str(n) + r'$M_{\odot}$'
#ax.annotate(s, (sizes_p.value_in(units.au)[i], m_mjup_p[i]), xytext=(sizes_p.value_in(units.au)[i]*1.02, m_mjup_p[i]),
#                    fontsize=14)


ax.axhline(y=(0.016 | units.MSun).value_in(units.MJupiter), c='r', ls=':')
ax.text(0.0, (0.016 | units.MSun).value_in(units.MJupiter) + 0.5, 'MMSN', fontsize=18, color='r')

ax.set_title('N=10, 1 bright star, 2 Myr')
ax.set_xlabel('Disk size [au]')
ax.set_ylabel(r'Disk mass [$M_{Jup}$]')
#pyplot.show()
fig.savefig('plot2.png')
