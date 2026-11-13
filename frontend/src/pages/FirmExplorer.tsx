import { useState } from "react";
import { api } from "../api/client";
import { Waterfall } from "../components/Waterfall";
import { formatMillions, formatPercent } from "../features/firms/format";
import { TARGET_LABEL } from "../features/replication/catalog";
import { useApi, useDebounced } from "../hooks/useApi";
import type { Firm, Peer, Target } from "../types/api";

const TARGETS: Target[] = ["lnm2b", "lnv2a", "lnv2s"];

function PeerTable({ title, peers }: { title: string; peers: Peer[] }) {
  return (
    <div>
      <h2>{title}</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Company</th>
              <th>Industry</th>
              <th className="num">Weight</th>
              <th className="num">Cumulative</th>
              <th className="num">ln multiple</th>
            </tr>
          </thead>
          <tbody>
            {peers.map((p) => (
              <tr key={p.peer_id}>
                <td>{p.name ?? p.peer_id}</td>
                <td>{p.industry ?? ""}</td>
                <td className="num">{p.weight.toFixed(3)}</td>
                <td className="num">{p.cumulative_weight.toFixed(2)}</td>
                <td className="num">{p.peer_log_multiple.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function FirmExplorer() {
  const [query, setQuery] = useState("");
  const [firm, setFirm] = useState<Firm | null>(null);
  const [date, setDate] = useState<string | undefined>(undefined);
  const [target, setTarget] = useState<Target>("lnm2b");
  const debounced = useDebounced(query);

  const results = useApi(() => api.firms(debounced, 30), [debounced]);
  const valuations = useApi(firm ? () => api.valuations(firm.id, date) : null, [firm?.id, date]);
  const month = valuations.data?.date;
  const shap = useApi(firm && month ? () => api.shap(firm.id, target, month) : null, [firm?.id, month, target]);
  const peers = useApi(firm && month ? () => api.peers(firm.id, target, month) : null, [firm?.id, month, target]);

  const pick = (f: Firm) => {
    setFirm(f);
    setDate(undefined);
  };

  return (
    <>
      <section className="card">
        <h2>Find a firm</h2>
        <input type="search" placeholder="Company name or RIC, for example Moderna or MRNA.OQ" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search firms" />
        {results.error && <p className="notice error">{results.error}</p>}
        <ul className="result-list">
          {(results.data ?? []).map((f) => (
            <li key={f.id}>
              <button aria-current={firm?.id === f.id} onClick={() => pick(f)}>
                <strong>{f.name ?? f.id}</strong> <span className="muted">{f.id} · {f.exchange ?? ""} · {f.industry ?? f.sector ?? ""}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      {firm && (
        <section className="card">
          <div className="card-head">
            <div>
              <h2>{firm.name ?? firm.id}</h2>
              <p className="muted small">Out-of-sample valuation by the three models. A negative error means the model values the firm below the market.</p>
            </div>
            {valuations.data && (
              <select value={valuations.data.date} onChange={(e) => setDate(e.target.value)} aria-label="Valuation month">
                {[...valuations.data.available_dates].reverse().map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            )}
          </div>
          {valuations.loading && <p className="notice">Loading…</p>}
          {valuations.error && <p className="notice">{valuations.error}</p>}
          {valuations.data && (
            <div className="grid-3">
              {valuations.data.valuations.map((v) => (
                <div key={v.target} className="card">
                  <div className="stat-label">{TARGET_LABEL[v.target]}</div>
                  <div className="stat-value">{formatPercent(v.valuation_error)}</div>
                  <div className="stat-sub">
                    valuation error · model {formatMillions(v.predicted_equity_value)} vs market {formatMillions(v.market_value)}
                  </div>
                  <div className="stat-sub">
                    {v.multiple}: predicted {v.predicted_multiple.toFixed(2)}, actual {v.actual_multiple.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {firm && month && (
        <section className="card">
          <div className="card-head">
            <div>
              <h2>Why this valuation, and who are the peers</h2>
              <p className="muted small">SHAP contributions to the predicted log multiple, and the training firms the prediction leans on.</p>
            </div>
            <div className="segmented" role="group" aria-label="Target">
              {TARGETS.map((t) => (
                <button key={t} aria-pressed={t === target} onClick={() => setTarget(t)}>
                  {t.replace("ln", "")}
                </button>
              ))}
            </div>
          </div>
          {shap.error && <p className="notice">{shap.error}</p>}
          {shap.data && <Waterfall items={shap.data.waterfall} />}
          {peers.error && <p className="notice">Peer weights: {peers.error}</p>}
          {peers.data && (
            <div className="grid-2">
              <PeerTable title="Most comparable" peers={peers.data.most_comparable} />
              <PeerTable title="Least comparable" peers={peers.data.least_comparable} />
            </div>
          )}
        </section>
      )}
    </>
  );
}
