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

## 🏋️‍♀️ Training + Monte Carlo

Below is a slightly more involved example of creating a model and deploying it for a Monte Carlo run.

This showcases two utility modules, namely `tce.training` and `tce.monte_carlo`. These mostly contain wrappers, so
feel free to avoid them! If you are using this for a novel research idea, it is likely that these wrappers are too
basic (which is a good thing for you!).

The first script is training a CuNi model using an EAM potential from Fischer et al.
(paper [here](https://doi.org/10.1016/j.actamat.2019.06.027)). In this script, we generate a bunch of random CuNi
solid solutions, attach an `ase.calculators.eam.EAM` calculator to each configuration, compute their energies, and
then train using the `tce.training.TrainingMethod.fit` method, which returns a `tce.training.CEModel` instance. The
container is then saved to be used for later.

**IMPORTANT**: These are unrelaxed energies! A real production environment should optimize the structure - see the
prior example on how to do this within a LAMMPS calculator.

```py
.. include:: ../examples/0-copper-nickel-training.py
```

The next script uses the saved container to run a canonical Monte Carlo simulation on a $10\times 10\times 10$
supercell, storing the configuration (saved in an `ase.Atoms` object) every 100 frames. We also set up a `logging`
configuration here, which will tell you how far-along the simulation is. Note that `trajectory` looks complicated, but
is just a list of `ase.Atoms` objects, so you have a lot of freedom to do what you wish with this trajectory later.

```py
.. include:: ../examples/1-copper-nickel-mc.py
```

These are then visualizable with a number of softwares, including [OVITO](https://www.ovito.org/). We can now also use
OVITO's Python library [here](https://pypi.org/p/ovito) and any of its plugins to do some analysis, as if our files
are from any other atomistic simulation software. Below we'll compute the Cowley short range order parameter using the
`cowley-sro-parameters` plugin [here](https://pypi.org/p/cowley-sro-parameters) (shameless plug... I'm the author 🙂).

```py
.. include:: ../examples/2-copper-nickel-sro.py
```

This generates the plot below. A negative value indicates attraction between two atom types. So, the solution is
clearly not fully random! We probably need a lot more than 10,000 steps too - this curve should bottom out once we
reach steady state. Note we can also just grab the potential energy from the `ase.Atoms` instances - the Monte Carlo
run stores this information using `ase.calculators.singlepoint.SinglePointCalculator` instances.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png"
    width=100%
    alt="CuNi SRO parameter from CE"
    title="SRO parameter"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png)

## 💻 Custom Training (Advanced)

Below is an example of using a custom training method to train the CE model. There are many reasons one might want to do
this. The example below is a very typical one - using [lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)). This
regularization technique minimizes the loss:

$$ L(\beta\; |\;\lambda) = \|X\beta - y\|_2^2 + \lambda \|\beta\|_1^2 $$

which better-supports sparse best-fit parameters $\hat{\beta}$, which may be useful if you only want to exclude
non-important clusters. We'll use `scikit-learn`'s interface for providing a model. You can really use any linear
model here (without an intercept...), see `scikit-learn`'s docs
[here](https://scikit-learn.org/stable/modules/linear_model.html) for more examples of these.

```py
.. include:: ../examples/3-sklearn-fitting.py
```

This script (it will be quite slow...) will calculate the number of nonzero cluster interaction coefficients as a
function of the regularization parameter. For larger regularization parameters, the number of nonzero coefficients
should decrease.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png"
    width=100%
    alt="Lasso regularization"
    title="Lasso"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png)


## 🧲 Learning a tensorial property

In general, one might also want to learn tensorial properties. This can be done by vectorizing the property in some
way, like [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation):

$$ \sigma = (\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy}) $$

Below is an example of changing the target property to stress rather than energy. It also showcases an important point
about `tce-lib`: our feature vectors are **extensive**, not intensive like other CE libraries. This matters when
training on intensive properties, like stress. Here, we need to actually train on an "extensive" version of stress, and
we can just make it intensive later. Of course, it is also fine to use this same pattern to train a CE model on
other scalar properties.

```py
.. include:: ../examples/4-tensorial-property.py
```

## 🔔 Callback functionality

The `tce.monte_carlo.monte_carlo` routine also has a `callback` argument that lets you inject a notification system
into the Monte Carlo run. This argument needs to be a function with signature:

```py
def callback(step: int, num_steps: int) -> None:
    ...
```

If it is not provided, it defaults to calling the `logging` library:

```py
import logging

LOGGER = logging.getLogger(__name__)

def callback(step_: int, num_steps_: int):
    LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")
```

But, you can do very cool things with this. It's a bit of a cute example that might not be practical, but you can
send notifications to third party systems like [Discord](https://en.wikipedia.org/wiki/Discord) using webhooks.

```py
.. include:: ../examples/5-callbacking.py
```

which will send a notification in whatever Discord channel once the MC run is finished. See
[Discord's documentation on webhooks](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) for
a tutorial on how to set up your own webhook URL. You can get really creative here too, like Slack's similar
functionality [here](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/), or the
[Gmail API](https://developers.google.com/workspace/gmail/api/guides). None of these are particularly useful for what I
have done above (sending a single email once the run is finished), but really shine for long runs where you want to
be periodicially notified.
"""

__version__ = "0.2.4"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

import warnings

from . import constants as constants
from . import structures as structures
from . import topology as topology


if __version__.startswith("0."):
    warnings.simplefilter("once", UserWarning)

    warnings.warn(
        f"{__name__} is in alpha. APIs are unstable and may change without notice. "
        f"Please report any problems at {__url__}/issues",
        UserWarning,
        stacklevel=2,
    )