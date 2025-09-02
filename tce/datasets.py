from dataclasses import dataclass, fields
from pathlib import Path
import json
from importlib.resources import files

from ase import Atoms, io

from tce.constants import LatticeStructure


DATASET_DIR = files("tce") / "datasets"


@dataclass
class Dataset:

    lattice_parameter: float
    lattice_structure: LatticeStructure
    description: str
    contact_info: str
    configurations: list[Atoms]

    def __repr__(self):
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "configurations":
                parts.append(f"{f.name}=[...]")
            else:
                parts.append(f"{f.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def from_dir(cls, directory: Path) -> "Dataset":

        with (DATASET_DIR / directory / "metadata.json").open("r") as file:
            metadata = json.load(file)

        metadata["lattice_structure"] = getattr(LatticeStructure, metadata["lattice_structure"].upper())

        return cls(
            **metadata,
            configurations=[
                io.read(path) for path in (DATASET_DIR / directory).glob("*.xyz")
            ]
        )
    

def available_datasets() -> list[str]:

    return list(x.name for x in DATASET_DIR.iterdir())
