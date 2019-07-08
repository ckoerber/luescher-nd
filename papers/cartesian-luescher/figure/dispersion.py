#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def nstep1(pe):
    return 2 - 2 * np.cos(pe)

def nstep2(pe):
    return 5/2 - 8/3 * np.cos(pe) + 1/6 * np.cos(2*pe)

def nstep3(pe):
    return 49/18 - 3 * np.cos(pe) + 3/10 * np.cos(2*pe) - 1/45 * np.cos(3*pe)

def nstep4(pe):
    return 205/72 - 16/5 * np.cos(pe) + 2/5 * np.cos(2*pe) - 16/315 * np.cos(3*pe) + 1/280 * np.cos(4*pe)

def nstepinf(pe):
    return pe**2

dispersion = {
    -1: nstepinf,
     1: nstep1,
     2: nstep2,
     3: nstep3,
     4: nstep4,
}

label = {
    -1: r'$n_{\mathrm{step}}=\infty$',
     1: r'$n_{\mathrm{step}}=1$',
     2: r'$n_{\mathrm{step}}=2$',
     3: r'$n_{\mathrm{step}}=3$',
     4: r'$n_{\mathrm{step}}=4$',
}

fig = plt.figure()
ax  = fig.add_subplot(111)

ax.set_title("Lattice Dispersion Relations")
ax.set_xlabel(r'$p\epsilon$')
ax.set_ylabel(r'$2 \mu E \epsilon^2$')

# ticks = 9
# divisions = np.arange(-ticks//2,1+ticks//2)
# xticks = 2*np.pi*divisions/(ticks-1)
# xticklabels = [r'$\frac{{ {n} \pi}}{{{denom}}}$'.format(n=2*n, denom=ticks-1) for n in divisions]

xticks = np.pi*np.array([-1,-1/2,0,1/2,1])
xticklabels = np.array([r'$-\pi$', r'$-\frac{\pi}{2}$',r'$0$', r'$+\frac{\pi}{2}$', r'$+\pi$',])

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)


pe = np.linspace(-np.pi, +np.pi, 1000)

for d in dispersion:
    ax.plot(pe, dispersion[d](pe), label=label[d])

ax.legend()

plt.savefig('./dispersion.pdf')
