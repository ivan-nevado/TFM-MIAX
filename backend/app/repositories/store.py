"""Read access to the replication outputs. Hides where the data live: a local folder or S3
(any URI pandas/fsspec understands). Services never build paths or touch files themselves."""

from __future__ import annotations

import fsspec
import pandas as pd

from app.core.errors import NotAvailableError, NotFoundError
from app.core.settings import Settings

TARGETS = ("lnm2b", "lnv2a", "lnv2s")


def _join(*parts: str) -> str:
    return "/".join(str(p).strip("/") if i else str(p).rstrip("/") for i, p in enumerate(parts))


class OutputStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.tables_dir = _join(settings.reports_root, settings.run_name, "tables")
        self.pred_dir = _join(settings.data_root, "predictions", settings.run_name)
        self.art_dir = _join(settings.data_root, "artifacts", settings.run_name)
        self.processed = _join(settings.data_root, "processed")
        self._firms: pd.DataFrame | None = None

    # ---- low level -------------------------------------------------------------------------
    @staticmethod
    def _exists(uri: str) -> bool:
        fs, path = fsspec.core.url_to_fs(uri)
        return fs.exists(path)

    @staticmethod
    def _glob(pattern: str) -> list[str]:
        fs, path = fsspec.core.url_to_fs(pattern)
        proto = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        return sorted(p if proto in ("file", "local") else f"{proto}://{p}" for p in fs.glob(path))

    def _parquet(self, uri: str, what: str, **kwargs) -> pd.DataFrame:
        if not self._exists(uri):
            raise NotAvailableError(f"{what} has not been produced yet ({uri})")
        return pd.read_parquet(uri, **kwargs)

    # ---- replication tables ------------------------------------------------------------------
    def list_tables(self) -> list[str]:
        files = self._glob(_join(self.tables_dir, "*.csv"))
        return [p.replace("\\", "/").rsplit("/", 1)[-1][:-4] for p in files]

    def table(self, table_id: str) -> pd.DataFrame:
        if not table_id.replace("_", "").isalnum():
            raise NotFoundError(f"unknown table {table_id!r}")
        uri = _join(self.tables_dir, f"{table_id}.csv")
        if not self._exists(uri):
            raise NotFoundError(f"table {table_id!r} not found")
        return pd.read_csv(uri)

    # ---- firms -------------------------------------------------------------------------------
    def firms(self) -> pd.DataFrame:
        if self._firms is None:
            self._firms = self._parquet(_join(self.processed, f"firms_{self.s.source}.parquet"), "firm table")
        return self._firms

    # ---- per firm-month outputs ----------------------------------------------------------------
    def _year_file(self, base: str, model: str, year: int) -> str:
        return _join(base, f"model={model}", f"seed={self.s.headline_seed}", f"year={year}.parquet")

    def predictions(self, target: str, firm_id: str, year: int | None = None) -> pd.DataFrame:
        model = f"ML_{target}"
        if year is not None:
            files = [self._year_file(self.pred_dir, model, year)]
        else:
            files = self._glob(
                _join(self.pred_dir, f"model={model}", f"seed={self.s.headline_seed}", "year=*.parquet")
            )
        frames = [pd.read_parquet(f, filters=[("id", "==", firm_id)]) for f in files if self._exists(f)]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date")

    def shap(self, target: str, firm_id: str, date: pd.Timestamp) -> pd.DataFrame:
        uri = self._year_file(_join(self.art_dir, "shap"), f"ML_{target}", date.year)
        df = self._parquet(uri, "SHAP values", filters=[("id", "==", firm_id)])
        return df[df["date"] == date]

    def peers(self, target: str, firm_id: str, date: pd.Timestamp) -> pd.DataFrame:
        uri = _join(self.art_dir, "peers", f"peers_top_{target}_{date:%Y%m}.parquet")
        return self._parquet(uri, f"peer weights for {date:%Y-%m}", filters=[("id", "==", firm_id)])

    def peer_months(self) -> list[str]:
        files = self._glob(_join(self.art_dir, "peers", "peers_top_lnm2b_*.parquet"))
        return sorted({f.rsplit("_", 1)[-1][:6] for f in files})
