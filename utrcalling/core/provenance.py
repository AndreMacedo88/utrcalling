"""How a count matrix was made, recorded in its ``.uns["utrcalling"]``.

Every command that writes a matrix records the utrcalling version, its command
line after those of the commands that made its input, and what the matrix holds.
The QC commands read the record to know what kind of matrix they describe.
"""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING, Any

from utrcalling._metadata import __version__

if TYPE_CHECKING:
    import anndata

KEY = "utrcalling"
# What a matrix holds: molecule counts per UTR size (raw sizes) or per site
# (peaks), each site's share of its UTR's molecules, or each UTR's mean size.
MATRICES = ("raw_sizes", "peaks", "fractions", "mean_sizes")


def command_line() -> str:
    """The command line of this process, as typed: ``utrcalling run ...``."""
    return shlex.join(["utrcalling", *sys.argv[1:]])


def record(
    adata: anndata.AnnData,
    source: anndata.AnnData | None = None,
    *,
    matrix: str | None = None,
    **details: Any,
) -> None:
    """Record in ``adata.uns["utrcalling"]`` how *adata* was made.

    Continues the record of *source*: keeps what it says, adds this process's
    command line to its commands, and sets the version to this one.

    Args:
        adata: The matrix about to be written.
        source: The matrix *adata* was made from (default: *adata* itself).
        matrix: What *adata* holds, one of MATRICES, if it differs from *source*.
        details: Settings to record, such as ``bandwidth=12``. None values are
            left out: h5ad files cannot store them.

    Raises:
        ValueError: If *matrix* is not one of MATRICES.
    """
    previous = recorded(adata if source is None else source)
    if matrix is not None and matrix not in MATRICES:
        raise ValueError(f"Unknown matrix {matrix!r}: expected one of {', '.join(MATRICES)}")
    info = previous | {k: v for k, v in details.items() if v is not None}
    info["version"] = __version__
    info["commands"] = [*previous.get("commands", []), command_line()]
    if matrix is not None:
        info["matrix"] = matrix
    adata.uns[KEY] = info


def recorded(adata: anndata.AnnData) -> dict[str, Any]:
    """What ``adata.uns["utrcalling"]`` records (empty for other matrices).

    The commands come as a list of strings (h5ad files store them as an array).
    """
    info = dict(adata.uns.get(KEY, {}))
    if "commands" in info:
        info["commands"] = [str(command) for command in info["commands"]]
    return info
