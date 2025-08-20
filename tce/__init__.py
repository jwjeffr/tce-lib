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

## 🔩 FeCr + EAM (basic)

Below is a very basic example of computing a best-fit interaction vector from LAMMPS data. We use LAMMPS and an EAM
potential from Eich et al. (paper [here](https://doi.org/10.1016/j.commatsci.2015.03.047)), use `tce-lib` to build a
best-fit interaction vector from a sequence of random samples, and cross-validate the results using `scikit-learn`.

```py
.. include:: ../examples/iron-chrome-lammps.py
```

This generates the plot below:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cross-val.png"
    width=100%
    alt="Residual errors during cross-validation"
    title="Residual errors"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cross-val.png)

The errors are not great here (a good absolute error is on the order of 1-10 meV/atom as a rule of thumb). The fit
would be much better if we included partially ordered samples as well. We emphasize that this is a very basic example,
and that a real production fit should be done against a more diverse training set than just purely random samples.

This example serves as a good template for using programs other than LAMMPS to compute energies. For example, one could
define a constructor that creates a `Calculator` instance that wraps VASP:

```py
from ase.calculators.vasp import Vasp

calculator_constructor = lambda: Vasp(
    prec="Accurate",
    encut=500,
    istart=0,
    ismear=1,
    sigma=0.1,
    nsw=400,
    nelmin=5,
    nelm=100,
    ibrion=1,
    potim=0.5,
    isif=3,
    isym=2,
    ediff=1e-5,
    ediffg=-5e-4,
    lreal=False,
    lwave=False,
    lcharg=False
)
```

See ASE's documentation [here](https://ase-lib.org/ase/calculators/vasp.html) for how to properly set this up!
"""

__version__ = "0.0.4"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

from . import constants as constants
from . import structures as structures
from . import topology as topology
