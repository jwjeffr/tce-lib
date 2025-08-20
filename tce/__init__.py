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

## 💎 Exotic Lattice Structures

Below is an example of injecting a custom lattice structure into `tce-lib`. To do this, we must extend the
`LatticeStructure` class, which we will do using [aenum](https://pypi.org/p/aenum/) (version `aenum==3.1.16`
specifically). We use a cubic diamond structure here as an example, but this extends to any atomic basis in any
tetragonal unit cell.

```py
.. include:: ../examples/exotic-lattice.py
```

We are also more than happy to include new lattice types as native options in `tce-lib`! Please either open an issue
[here](https://github.com/MUEXLY/tce-lib/issues), or a pull request [here](https://github.com/MUEXLY/tce-lib/pulls) if
you are familiar with GitHub.
"""

__version__ = "0.0.3"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

from . import constants as constants
from . import structures as structures
from . import topology as topology
