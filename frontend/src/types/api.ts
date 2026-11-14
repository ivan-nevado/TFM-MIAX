// Types of the backend API (backend/app/schemas/models.py).

export type Target = "lnm2b" | "lnv2a" | "lnv2s";

export type Cell = string | number | boolean | null;

export interface TableResponse {
  table_id: string;
  columns: string[];
  rows: Record<string, Cell>[];
}

export interface Firm {
  id: string;
  name: string | null;
  exchange: string | null;
  sector: string | null;
  industry: string | null;
}

export interface Valuation {
  id: string;
  date: string;
  target: Target;
  multiple: string;
  actual_log_multiple: number;
  predicted_log_multiple: number;
  actual_multiple: number;
  predicted_multiple: number;
  market_value: number;
  predicted_equity_value: number;
  valuation_error: number;
  block: number;
  n_train: number;
  trees: number | null;
}

export interface ValuationSet {
  id: string;
  name: string | null;
  date: string;
  available_dates: string[];
  valuations: Valuation[];
}

export interface ShapItem {
  label: string;
  value: number;
  kind: "base" | "contribution" | "total";
  category: string | null;
}

export interface ShapResponse {
  id: string;
  date: string;
  target: Target;
  base_value: number;
  prediction: number;
  waterfall: ShapItem[];
}

export interface Peer {
  peer_id: string;
  name: string | null;
  industry: string | null;
  weight: number;
  cumulative_weight: number;
  rank: number;
  peer_log_multiple: number;
}

export interface PeersResponse {
  id: string;
  date: string;
  target: Target;
  n_peers: number;
  most_comparable: Peer[];
  least_comparable: Peer[];
}
