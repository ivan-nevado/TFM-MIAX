"""WRDS adapter (CRSP monthly + Compustat quarterly): specified, not implemented.

This is the adapter that would rerun the identical pipeline on the paper's own data and period
(set ``source: wrds`` and ``period.start: "1980-01"`` in the config). Nothing else in the
repository needs to change. It is a stub because the project has no WRDS subscription.

Files expected under ``data/raw/wrds/`` (CSV or parquet exports of the WRDS tables):

``crsp_msf``        crsp.msf joined with crsp.msenames: permno, date, ret, altprc, shrout, shrcd,
                    exchcd, siccd, and crsp.msedelist: dlret (delisting return).
``comp_fundq``      comp.fundq (indfmt INDL, datafmt STD, popsrc D, consol C): gvkey, datadate,
                    fyearq, fqtr and the quarterly items listed below.
``ccm_link``        crsp.ccmxpf_lnkhist: gvkey, lpermno, linktype (LU, LC), linkprim (P, C),
                    linkdt, linkenddt.

Mapping to :class:`relval.data.schemas.RawTables`:

* ``id``            permno as a string (the paper counts firms by gvkey; keep gvkey for reporting).
* ``market.mv``     ``abs(altprc) * shrout / 1000`` (shrout is in thousands) -> USD millions.
* ``market.ret``    ``ret`` compounded with ``dlret`` in the delisting month.
* ``in_universe``   ``shrcd in (10, 11) and exchcd in (1, 2, 3)``, evaluated month by month.
* ``nyse``          ``exchcd == 1``, month by month (historical exchange, unlike the LSEG proxy).
* ``static.sic``    CRSP ``siccd`` (historical; take the latest for the static table or extend the
                    panel builder to use a monthly SIC column if exact history is wanted).
* Fundamentals      Compustat items carry a ``q`` suffix (atq, actq, lctq, ltq, cheq, invtq,
                    rectq, apq, ppentq, dlttq, dlcq, pstkq, ceqq, txditcq, mibq, saleq, cogsq,
                    xsgaq, xrdq, xintq, oiadpq -> ebit, dpq, piq, txtq, ibq, dvpq -> prefdiv,
                    ibadjq, miiq, nopiq, spiq). Already in USD millions.
* Cash-flow items   ``oancfy``, ``capxy``, ``dvy`` are fiscal-year-to-date in Compustat: convert to
                    single-quarter amounts (value minus the previous quarter's value within the
                    fiscal year; the first fiscal quarter is taken as is) -> ocf, capx, dv.
* ``fund_a``        return an empty frame with the standard columns: the paper uses quarterly
                    statements only, so the annual fallback is never triggered.

LSEG-specific definition switches must then be turned off in the config
(``zero_fill_items: []``, ``opcost_from_ebitda: false``, ``dp_adjustment: none``).
"""

from __future__ import annotations

from pathlib import Path

from relval.data.schemas import RawTables

EXPECTED_FILES = ("crsp_msf", "comp_fundq", "ccm_link")


class WrdsAdapter:
    name = "wrds"

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def load(self) -> RawTables:
        present = [p.stem for p in self.dir.glob("*") if p.stem in EXPECTED_FILES]
        raise NotImplementedError(
            "The WRDS adapter is a documented stub. Place the CRSP/Compustat exports "
            f"{EXPECTED_FILES} in {self.dir} (found: {present or 'none'}) and implement the mapping "
            "described in relval/data/wrds.py. See docs/replication/BLOCKERS.md."
        )
