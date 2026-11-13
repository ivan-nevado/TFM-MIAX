import { useState } from "react";
import { FirmExplorer } from "./pages/FirmExplorer";
import { ReplicationDashboard } from "./pages/ReplicationDashboard";

type Page = "replication" | "firms";

const PAGES: { id: Page; label: string }[] = [
  { id: "replication", label: "Replication" },
  { id: "firms", label: "Firm explorer" },
];

export function App() {
  const [page, setPage] = useState<Page>(window.location.hash === "#firms" ? "firms" : "replication");
  const go = (p: Page) => {
    window.location.hash = p;
    setPage(p);
  };
  return (
    <>
      <header className="app-header">
        <div>
          <h1>Relative valuation with machine learning</h1>
          <p className="muted small">Replication of Geertsema &amp; Lu (2023) on LSEG data, 1990-2019</p>
        </div>
        <nav className="tabs" role="tablist">
          {PAGES.map((p) => (
            <button key={p.id} role="tab" aria-selected={page === p.id} onClick={() => go(p.id)}>
              {p.label}
            </button>
          ))}
        </nav>
      </header>
      <main>{page === "replication" ? <ReplicationDashboard /> : <FirmExplorer />}</main>
    </>
  );
}
