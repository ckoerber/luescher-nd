#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import finalize
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import LINE_STYLE

psq = 15.34824844488746404710  # Counterterm for the FFT method, from Mathematica.

# Data, from dispersion.nb
counterterms = np.array(
    [
        [1, 19.953112135459566],
        [2, 17.292009589188943],
        [3, 16.52764526114633],
        [4, 16.180080105565406],
        [6, 15.8613372773524],
        [8, 15.714730988261573],
        [12, 15.57818596398218],
        [16, 15.514255072974349],
        [24, 15.453701476619367],
        [32, 15.424877298562093],
        [48, 15.397185274010763],
        [64, 15.38382809082002],
        [96, 15.370854042506616],
        [128, 15.36453292608324],
        [192, 15.358342876873563],
    ]
)


setup(n_colors=1, pgf=True, rc_kwargs={"pgf.preamble": r"\usepackage{amsmath, amssymb}"})

fig = plt.figure(figsize=(6, 4))
plt.minorticks_off()

fig, ax = plt.subplots(2, sharex=True, sharey=False)

DASHED = dict(LINE_STYLE)
DASHED["ls"] = "dashed"

ax[0].axhline(y=psq, color="k", **DASHED)
ax[0].plot(*counterterms.T, "o", ms=2)
ax[0].set_xscale("log")
ax[0].set_ylabel(r"$\mathcal{L}^{\boxplus}_3$")
ax[0].set_ylim(15, 21)

ax[1].errorbar(
    counterterms[:, 0], [1.0e-3] * len(counterterms[:, 0]), counterterms[:, 1] / psq - 1
)
ax[1].set_ylim(1.0e-3, 5.0e-1)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$n_{\mathrm{step}}$")
ax[1].set_ylabel(r"$\frac{\mathcal{L}^{\boxplus}_3}{\mathcal{L}^{\Box}_3} - 1$")


pow2 = [1, 2, 4, 8, 16, 32, 64, 128]
for a in ax:
    a.set_xticks(pow2)
    a.set_xticklabels(pow2)
    a.minorticks_off()

finalize(fig)

# plt.savefig('./counterterm-nstep.png', bbox_inches='tight', dpi=300)
# plt.show()
fig.savefig("./counterterm-nstep.pgf", **EXPORT_OPTIONS)
