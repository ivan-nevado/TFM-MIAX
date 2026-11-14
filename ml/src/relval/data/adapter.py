"""Source adapter interface.

The pipeline never reads vendor files directly. An adapter converts a vendor download into the
standardised :class:`~relval.data.schemas.RawTables`; everything downstream (panel, models,
evaluation) is source-agnostic. Running the paper's own data (CRSP/Compustat, 1980-2019) therefore
only needs the ``wrds`` adapter and ``source: wrds`` in the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from relval.data.schemas import RawTables


class SourceAdapter(Protocol):
    name: str

    def load(self) -> RawTables:
        """Read the vendor files and return the standardised tables."""
        ...


def get_adapter(source: str, raw_dir: Path) -> SourceAdapter:
    if source == "lseg":
        from relval.data.lseg import LsegAdapter

        return LsegAdapter(raw_dir / "lseg")
    if source == "wrds":
        from relval.data.wrds import WrdsAdapter

        return WrdsAdapter(raw_dir / "wrds")
    raise ValueError(f"unknown source adapter: {source!r}")
