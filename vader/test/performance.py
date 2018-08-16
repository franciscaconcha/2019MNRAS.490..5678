"""
This script runs some automated performance tests.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import numpy.ma as ma
from matplotlib.patches import Rectangle

# Set of Anderson M values, and set of problems to run
aa_m_vals = ['0', '1', '2', '4', '8', '16']
probs = ['selfsim', 'ring', 'gidisk', 'ringrad']
names = ['Self-similar', 'Ring', 'GI Disk', 'Rad. Ring']
logdt = ['-2.5', '-6', '-3.5', '-7.5']
max_iter = 100
max_iter1 = 40
max_step = 1000

# Set up path
cwd = os.path.split(os.getcwd())[-1]
if cwd == 'vader':
    _path = '.'
else:
    _path = '..'

# Run the tests
for i, p in enumerate(probs):
    for aa_m in aa_m_vals:
        cmdstr = sys.executable + ' ' + \
                 os.path.join(_path, 'test', 'performance_sub.py') + \
                 ' ' + p + ' ' + aa_m + ' ' + logdt[i]
        os.system(cmdstr)

# Read the output files
nIterCN = np.zeros((len(probs), len(aa_m_vals)), dtype='int')
nIterBE = np.zeros((len(probs), len(aa_m_vals)), dtype='int')
residCN = ma.masked_array(
    np.zeros((len(probs), len(aa_m_vals), max_iter)), mask=[False])
residBE = ma.masked_array(
    np.zeros((len(probs), len(aa_m_vals), max_iter)), mask=[False])
rtypeCN = ma.masked_array(
    np.zeros((len(probs), len(aa_m_vals), max_iter), dtype='int'), mask=[False])
rtypeBE = ma.masked_array(
    np.zeros((len(probs), len(aa_m_vals), max_iter), dtype='int'), mask=[False])
tFin = np.zeros((2, len(probs), len(aa_m_vals)))
nIter = np.zeros((2, len(probs), len(aa_m_vals)), dtype='int')
nStep = np.zeros((2, len(probs), len(aa_m_vals)), dtype='int')
nFail = np.zeros((2, len(probs), len(aa_m_vals)), dtype='int')
residSum = np.zeros((2, len(probs), len(aa_m_vals), max_iter1))
iterStep = np.zeros((2, len(probs), len(aa_m_vals), max_step), dtype='int')
timeStep = np.zeros((2, len(probs), len(aa_m_vals), 3))
timeSim = np.zeros((2, len(probs), len(aa_m_vals), 4))
dirname = os.path.join(_path, 'output')
for i, p in enumerate(probs):
    for j, aa_m in enumerate(aa_m_vals):
        fname = p+'_'+aa_m+'_'+logdt[i]+'.txt'
        fullname = os.path.join(dirname, fname)
        fp = open(fullname, 'r')
        line = fp.readline()
        for k in range(3):
            timeStep[0,i,j,k] = float(line.split()[k])
        for k in range(3):
            timeStep[1,i,j,k] = float(line.split()[k+3])
        line = fp.readline()
        nIterCN[i,j] = int(line.split()[0])
        nIterBE[i,j] = int(line.split()[1])
        for k in range(max_iter):
            line = fp.readline()
            residCN[i,j,k] = float(line.split()[0])
            rtypeCN[i,j,k] = int(line.split()[1])
            residBE[i,j,k] = float(line.split()[2])
            rtypeCN[i,j,k] = int(line.split()[3])
        residCN.mask[i,j,nIterCN[i,j]:]=True
        residBE.mask[i,j,nIterBE[i,j]:]=True
        rtypeCN.mask[i,j,nIterCN[i,j]:]=True
        rtypeBE.mask[i,j,nIterBE[i,j]:]=True
        line = fp.readline()
        for k in range(4):
            timeSim[0,i,j,k] = float(line.split()[k])
        for k in range(4):
            timeSim[1,i,j,k] = float(line.split()[k+4])
        line = fp.readline()
        tFin[0,i,j] = float(line.split()[0])
        nStep[0,i,j] = int(line.split()[1])
        nFail[0,i,j] = int(line.split()[2])
        nIter[0,i,j] = long(line.split()[3])
        tFin[1,i,j] = float(line.split()[4])
        nStep[1,i,j] = int(line.split()[5])
        nFail[1,i,j] = int(line.split()[6])
        nIter[1,i,j] = long(line.split()[7])
        for k in range(max_iter1):
            line = fp.readline()
            residSum[0,i,j,k] = float(line.split()[0])
            residSum[1,i,j,k] = float(line.split()[1])
        for k in range(max_step):
            line = fp.readline()
            iterStep[0,i,j,k] = int(line.split()[0])
            iterStep[1,i,j,k] = int(line.split()[1])
        fp.close()

# Make plot of single time step results
idx=np.arange(max_iter)+1
colors=['r', 'g', 'b', 'c', 'm', 'k']
plt.figure(figsize=(8,8))

for i in range(len(names)):
    ax=plt.subplot(2,2,i+1)
    plotsCN=[]
    plotsBE=[]
    labels=[]
    for j, aa_m in enumerate(aa_m_vals):
        pCN,=plt.plot(idx, np.abs(residCN[i,j,:]), colors[j], lw=2)
        pBE,=plt.plot(idx, np.abs(residBE[i,j,:]), colors[j]+'--')
        plotsCN.append(pCN)
        plotsBE.append(pBE)
        labels.append(r'$M = '+aa_m+'$')
    plt.yscale('log')
    if i % 2 == 0:
        plt.ylabel('Residual')
        plt.xlim([0,99])
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlim([0,100])
    if i > 1:
        plt.xlabel('Iteration number')
        plt.ylim([1e-10,9e2])
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylim([1e-10,1e3])
    leg1=plt.legend([], [], title=names[i], loc='upper left')
    if (i==1):
        leg2=plt.legend(plotsCN, labels, title='Crank-Nicolson', 
                        loc='upper right')
        ax.add_artist(leg1)
    elif (i==3):
        leg2=plt.legend(plotsBE, labels, title='Backwards Euler', 
                        loc='upper right')
        ax.add_artist(leg1)

plt.subplots_adjust(wspace=0.0, hspace=0.0)

print("Writing output/performance1.pdf.")
fname = os.path.join(dirname, 'performance1')
plt.savefig(fname+'.pdf')

# Make plots of full simulation results
plt.figure(figsize=(8,8))
aa_m = np.array(aa_m_vals, dtype='int')
for i in range(len(names)):
    ax=plt.subplot(2,2,i+1)
    tnormCN = timeSim[0,i,0,0]/tFin[0,i,0]
    tnormBE = timeSim[1,i,0,0]/tFin[1,i,0]
    plots=[]
    labels=[]
    p1,=plt.plot(aa_m+1, timeSim[0,i,:,0]/tFin[0,i,:]/tnormCN, 
                 'k', lw=2)
    plots.append(p1)
    labels.append('Total')
    plt.fill_between(aa_m+1, timeSim[0,i,:,3]/tFin[0,i,:]/tnormCN,
                     np.zeros(len(aa_m)), color='g', alpha=0.5)
    r1 = Rectangle((0, 0), 1, 1, color='g', alpha=0.5)
    plots.append(r1)
    labels.append(r'Problem-specific')
    plt.fill_between(aa_m+1, 
                     (timeSim[0,i,:,3]+timeSim[0,i,:,2])/tFin[0,i,:]/tnormCN,
                     timeSim[0,i,:,3]/tFin[0,i,:]/tnormCN,
                     color='b', alpha=0.5)
    r2 = Rectangle((0, 0), 1, 1, color='b', alpha=0.5)
    plots.append(r2)
    labels.append(r'Anderson')
    plt.fill_between(aa_m+1, 
                     timeSim[0,i,:,0]/tFin[0,i,:]/tnormCN,
                     (timeSim[0,i,:,3]+timeSim[0,i,:,2])/tFin[0,i,:]/tnormCN,
                     color='r', alpha=0.5)
    r3 = Rectangle((0, 0), 1, 1, color='r', alpha=0.5)
    plots.append(r3)
    labels.append(r'Advance')
    plt.plot(aa_m+1, (aa_m+1)*0+1.0, 'k--', alpha=0.25, lw=2)
    plt.plot(aa_m+1, (aa_m+1)*0+0.5, 'k--', alpha=0.25, lw=2)
    plt.plot(aa_m+1, (aa_m+1)*0+0.25, 'k--', alpha=0.25, lw=2)
    plt.plot(aa_m+1, (aa_m+1)*0+2.0, 'k--', alpha=0.25, lw=2)
    plt.xlim([1,16])
    plt.xscale('log')
    if i>1:
        plt.xlabel('$M$')
        plt.ylim([0, 1.39])
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylim([0,3])
    if i==1:
        leg1=plt.legend(plots, labels, loc='upper left')
    plt.legend([], [], title=names[i], loc='upper right')
    if i==1:
        ax.add_artist(leg1)
    if i % 2 == 0:
        plt.ylabel('Computational cost')
        tickloc=[float(m)+1 for m in aa_m_vals[:-1]]
        plt.xticks(tickloc, aa_m_vals[:-1])
        plt.tick_params(axis='x', which='minor', bottom='off', top='off')
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        tickloc=[float(m)+1 for m in aa_m_vals]
        plt.xticks(tickloc, aa_m_vals)
        plt.tick_params(axis='x', which='minor', bottom='off', top='off')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
print("Writing output/performance2.pdf.")
fname = os.path.join(dirname, 'performance2')
plt.savefig(fname+'.pdf')


# Print latex-formatted table of single time step results
print("Latex output (single step):\n")
print("\\begin{tabular}{ccrrrrrrrr}")
print("\hline\hline")
print("& & \multicolumn{2}{c}{Self-similar} & \multicolumn{2}{c}{Ring} & \multicolumn{2}{c}{GI Disk} & \\multicolumn{2}{c}{Rad.~ring} \\\\")
print(" & \multicolumn{1}{l}{$M$} & CN & BE & CN & BE & CN & BE & CN & BE \\\\ \hline \hline")
for i, m in enumerate(aa_m_vals):
    if i==0:
        tmp = "\multirow{"+str(len(aa_m_vals))+"}{*}{$N_{\\rm iter}$} & "
    else:
        tmp = "& "
    print(tmp + m+"$\ldots\ldots$ & " + ("{:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d} \\\\".format(nIterCN[0,i], nIterBE[0,i], nIterCN[1,i], nIterBE[1,i], nIterCN[2,i], nIterBE[2,i], nIterCN[3,i], nIterBE[3,i])))
print("\hline")
for i, m in enumerate(aa_m_vals):
    if i==0:
        tmp = "\multirow{"+str(len(aa_m_vals))+"}{*}{Time [ms]} & "
    else:
        tmp = "& "
    print(tmp + m +"$\ldots\ldots$ & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} \\\\".format(timeStep[0,0,i,0]*1e3, timeStep[1,0,i,0]*1e3, timeStep[0,1,i,0]*1e3, timeStep[1,1,i,0]*1e3, timeStep[0,2,i,0]*1e3, timeStep[1,2,i,0]*1e3, timeStep[0,3,i,0]*1e3, timeStep[1,3,i,0]*1e3))
print("\hline\hline")
print("\end{tabular}")
print("\n")


# Print latex-formatted table of multi-step results
print("Latex output (multi-step):\n")
print("\\begin{tabular}{ccrrrrrrrr}")
print("\hline\hline")
print("& & \multicolumn{2}{c}{Self-similar} & \multicolumn{2}{c}{Ring} & \multicolumn{2}{c}{GI Disk} & \\multicolumn{2}{c}{Rad.~ring} \\\\")
print(" & \multicolumn{1}{l}{$M$} & CN & BE & CN & BE & CN & BE & CN & BE \\\\ \hline \hline")
for i, m in enumerate(aa_m_vals):
    if i==0:
        tmp = "\multirow{"+str(len(aa_m_vals))+"}{*}{$N_{\\rm Step}$}\n& "
    else:
        tmp = "& "
    print(tmp + m+"$\ldots\ldots$ & " + ("{:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d} \\\\".format(nStep[0,0,i], nStep[1,0,i], nStep[0,1,i], nStep[1,1,i], nStep[0,2,i], nStep[1,2,i], nStep[0,3,i], nStep[1,3,i])))
print("\hline")
for i, m in enumerate(aa_m_vals):
    if i==0:
        tmp = "\multirow{"+str(len(aa_m_vals))+"}{*}{$N_{\\rm iter}$}\n& "
    else:
        tmp = "& "
    print(tmp + m+"$\ldots\ldots$ & " + ("{:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d}\qquad & {:d} & {:d} \\\\".format(nIter[0,0,i], nIter[1,0,i], nIter[0,1,i], nIter[1,1,i], nIter[0,2,i], nIter[1,2,i], nIter[0,3,i], nIter[1,3,i])))
print("\hline")
for i, m in enumerate(aa_m_vals):
    if i==0:
        tmp = "\multirow{"+str(len(aa_m_vals))+"}{*}{Time [s]}\n& "
    else:
        tmp = "& "
    print(tmp + m +"$\ldots\ldots$ & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} & {:5.2f} \\\\".format(timeSim[0,0,i,0], timeSim[1,0,i,0], timeSim[0,1,i,0], timeSim[1,1,i,0], timeSim[0,2,i,0], timeSim[1,2,i,0], timeSim[0,3,i,0], timeSim[1,3,i,0]))
print("\hline\hline")
print("\end{tabular}")
print("\n")
