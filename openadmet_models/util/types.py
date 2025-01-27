
from typing import TypeAlias
import os

# define a type for paths and PathLike objects
Pathy: TypeAlias = str | bytes | os.PathLike
