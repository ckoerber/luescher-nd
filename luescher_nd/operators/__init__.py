# pylint: disable=C0301
"""Operators

Naming conventions for operators (projectors):

The method `p = get_projector_to_{space}` returns an operator with the action
```
p |psi> = |psi>
```
if `|psi>` is contained in `{space}` and else zero.
"""
import numpy as np
import luescher_nd.lattice as lattice
import scipy.sparse as sp

import luescher_nd.operators.parity as parity
# Backwards compatibility:
from luescher_nd.operators.parity import operator as get_parity_operator
from luescher_nd.operators.parity import projector as get_projector_to_parity


import luescher_nd.operators.a1g as a1g
# Backwards compatibility:
from luescher_nd.operators.a1g import projector as get_projector_to_a1g
from luescher_nd.operators.a1g import complement as get_projector_to_not_a1g
from luescher_nd.operators.a1g import reducer as get_a1g_reducer
