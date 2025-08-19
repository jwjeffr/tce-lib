r"""
.. include:: ../README.md

# Examples

## ⚛️ Using Atomic Simulation Environment (ASE)

Below is an example of converting an `ase.Atoms` object into a feature vector $\mathbf{t}$. The mapping is not exactly
one-to-one, since an `ase.Atoms` object sits on a dynamic lattice rather than a static one, but we can regardless
provide `tce-lib` sufficient information to compute $\mathbf{t}$. The code snippet below uses the version `ase==3.26.0`.

```py
.. include:: ../examples/using-ase.py
```
"""

__version__ = "0.0.1"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

from . import constants as constants
from . import structures as structures
from . import topology as topology
