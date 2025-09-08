# tce-lib

[![Custom shields.io](https://img.shields.io/badge/docs-orange?logo=github&logoColor=green&label=gh-pages)](https://muexly.github.io/tce-lib)
[![Stable Version](https://img.shields.io/pypi/v/tce-lib?color=blue)](https://pypi.org/project/tce-lib/)
[![Static Badge](https://img.shields.io/badge/License-MIT-8A2BE2)](https://en.wikipedia.org/wiki/MIT_License)


[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Tested with pytest](https://img.shields.io/badge/pytest-tested-blue?logo=pytest)](https://docs.pytest.org/en/stable/)

<img src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/assets/logo.png" alt="tce-lib logo" style="width:50%;height:auto;">


## 🔎 What is tce-lib?

`tce-lib` is a library for creating and deploying tensor cluster expansion models of concentrated alloys following
our work on [arXiv](https://arxiv.org/abs/2509.04686). The core philosophy of `tce-lib` is to respect the 
[strategy pattern](https://en.wikipedia.org/wiki/Strategy_pattern) as core to the library's functionality. This design
pattern stages workflows as sequences of strategies, of which the user can override each. This allows for the majority 
of users to plug-and-play for an ordinary workflow, while still supporting fine-grained autonomy for more advanced 
users. 

## 📩 Installation

`tce-lib` is installable via the Python Package Index:

```shell
pip install tce-lib
```

or, from source:

```shell
git clone https://github.com/MUEXLY/tce-lib
pip install -e tce-lib/
```

## 📌 Citation

Please cite our work [here](https://arxiv.org/abs/2509.04686) if you use `tce-lib` in your work.

## 💙 Acknowledgements

Authors acknowledge support from the U.S. Department of Energy, Office of Basic Energy Sciences, Materials Science and Engineering Division under Award No. DE-SC0022980.

## 🐝 Found a bug?

Please open an issue [here](https://github.com/MUEXLY/tce/issues), with a description of the issue and a [minimal, reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) of the issue.

## 📑 License

`tce-lib` is released under the MIT license.
