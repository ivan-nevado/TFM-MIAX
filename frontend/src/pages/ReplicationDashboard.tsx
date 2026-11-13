import { useMemo, useState } from "react";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { LineChart, type Series } from "../components/LineChart";
import { StatTile } from "../components/StatTile";
import { FEATURED_TABLES, MULTIPLES, seriesForMultiple, type Multiple } from "../features/replication/catalog";
import { useApi } from "../hooks/useApi";
import type { TableResponse } from "../types/api";

function headlineTiles(table: TableResponse | null) {
  if (!table) return [];
  return ["ML_lnm2b", "ML_lnv2a", "ML_lnv2s"].flatMap((model) => {
    const row = table.rows.find((r) => r.model === model);
    if (!row || typeof row.med_abs_eps !== "number") return [];
    const paper = typeof row.med_abs_eps_paper === "number" ? row.med_abs_eps_paper.toFixed(2) : "n/a";
    return [{ label: `${model}: median absolute error`, value: row.med_abs_eps.toFixed(3), sub: `paper (1980-2019): ${paper}` }];
  });
}

export function ReplicationDashboard() {
  const [multiple, setMultiple] = useState<Multiple>("m2b");
  const [tableId, setTableId] = useState(FEATURED_TABLES[0].id);

  const available = useApi(() => api.tables(), []);
  const table2 = useApi(() => api.table("table2_vs_paper"), []);
  const monthly = useApi(() => api.table("fig1_monthly_median_abs_error"), []);
  const selected = useApi(() => api.table(tableId), [tableId]);

  const options = useMemo(() => {
    const ids = new Set(available.data ?? []);
    const featured = FEATURED_TABLES.filter((t) => ids.has(t.id));
    const rest = (available.data ?? []).filter((id) => !FEATURED_TABLES.some((t) => t.id === id)).map((id) => ({ id, title: id, note: "" }));
    return [...featured, ...rest];
  }, [available.data]);

  const chart = useMemo(() => {
    const rows = monthly.data?.rows ?? [];
    const dates = rows.map((r) => String(r.date));
    const series: Series[] = seriesForMultiple(multiple).map((s) => ({
      ...s,
      values: rows.map((r) => (typeof r[s.key] === "number" ? (r[s.key] as number) : null)),
    }));
    return { dates, series };
  }, [monthly.data, multiple]);

  const note = options.find((t) => t.id === tableId)?.note;
  const tiles = headlineTiles(table2.data);

  return (
    <>
      {available.error && (
        <div className="card">
          <p className="error">The API is not reachable or has no outputs yet: {available.error}</p>
          <p className="muted small">Run the pipeline (`relval replicate`) and start the backend.</p>
        </div>
      )}

      {tiles.length > 0 && (
        <div className="grid-3">
          {tiles.map((t) => (
            <StatTile key={t.label} {...t} />
          ))}
        </div>
      )}

      <section className="card">
        <div className="card-head">
          <div>
            <h2>Valuation error over time</h2>
            <p className="muted small">Median absolute valuation error by month, out-of-sample predictions. Lower is better.</p>
          </div>
          <div className="segmented" role="group" aria-label="Multiple">
            {MULTIPLES.map((m) => (
              <button key={m} aria-pressed={m === multiple} onClick={() => setMultiple(m)}>
                {m}
              </button>
            ))}
          </div>
        </div>
        {monthly.loading ? <p className="notice">Loading…</p> : <LineChart dates={chart.dates} series={chart.series} yLabel="Median absolute valuation error" />}
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <h2>Replication tables</h2>
            <p className="muted small">{note || "Every table produced by the pipeline, ours next to the paper where available."}</p>
          </div>
          <select value={tableId} onChange={(e) => setTableId(e.target.value)} aria-label="Table">
            {options.map((t) => (
              <option key={t.id} value={t.id}>
                {t.title}
              </option>
            ))}
          </select>
        </div>
        {selected.loading && <p className="notice">Loading…</p>}
        {selected.error && <p className="notice">This table has not been produced yet.</p>}
        {selected.data && <DataTable table={selected.data} />}
      </section>
    </>
  );
}
