import os
from typing import TypeAlias

# define a type for paths and PathLike objects
Pathy: TypeAlias = str | bytes | os.PathLike
