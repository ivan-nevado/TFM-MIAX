// Typed API client. Components never call fetch directly.
import type { Firm, PeersResponse, ShapResponse, TableResponse, Target, ValuationSet } from "../types/api";

const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function get<T>(path: string, params: Record<string, string | number | undefined> = {}): Promise<T> {
  const query = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  const response = await fetch(`${BASE}${path}${query ? `?${query}` : ""}`);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  tables: () => get<string[]>("/api/v1/replication/tables"),
  table: (id: string) => get<TableResponse>(`/api/v1/replication/tables/${encodeURIComponent(id)}`),
  firms: (query: string, limit = 20) => get<Firm[]>("/api/v1/firms", { query, limit }),
  valuations: (id: string, date?: string) =>
    get<ValuationSet>(`/api/v1/valuations/${encodeURIComponent(id)}`, { date }),
  shap: (id: string, target: Target, date?: string, top = 8) =>
    get<ShapResponse>(`/api/v1/valuations/${encodeURIComponent(id)}/shap`, { date, target, top }),
  peers: (id: string, target: Target, date?: string) =>
    get<PeersResponse>(`/api/v1/valuations/${encodeURIComponent(id)}/peers`, { date, target }),
};
