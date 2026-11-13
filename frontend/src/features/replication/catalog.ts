// Presentation metadata of the replication outputs: which tables the dashboard lists first, and
// which role (and therefore colour) each model plays in a chart. No numbers live here.

export interface TableEntry {
  id: string;
  title: string;
  note: string;
}

export const FEATURED_TABLES: TableEntry[] = [
  { id: "findings", title: "Qualitative findings", note: "The six findings of the paper that should hold on any sample." },
  { id: "thresholds", title: "Working thresholds", note: "Flagged when outside, never forced." },
  { id: "panel_table1", title: "Table 1: sample selection", note: "Ours (1990-2019, LSEG) next to the paper (1980-2019)." },
  { id: "table2_vs_paper", title: "Table 2: model performance", note: "Columns ending in _paper are the paper's." },
  { id: "table2_seed_spread", title: "Table 2: seed spread", note: "Spread of the base models across block seeds." },
  { id: "ia5_common_sample", title: "Table IA.5: common subsample", note: "Firm-months valued by every model." },
  { id: "table3_vs_paper", title: "Table 3: long-short portfolios", note: "Value-weighted, NYSE breakpoints, FF6 alphas." },
  { id: "table4_vs_paper", title: "Table 4: sources of the gains", note: "One change at a time on the same observations." },
  { id: "table7", title: "Table 7: top variables (SHAP)", note: "Scaled absolute SHAP, percentage points." },
  { id: "table8_vs_paper", title: "Table 8: category importance", note: "Sum of the variables of each category." },
  { id: "table9", title: "Table 9: importance over time", note: "Largest changes between the first and last decade." },
  { id: "table5", title: "Table 5: peers of the focus firm", note: "Most and least comparable firms by peer weight." },
  { id: "ia4_summary_statistics", title: "Table IA.4: input coverage", note: "Coverage, mean and median of the 97 inputs." },
  { id: "sensitivity_runs", title: "Sensitivity runs", note: "Industry numeric, listed-only, beta history." },
];

export const MULTIPLES = ["m2b", "v2a", "v2s"] as const;
export type Multiple = (typeof MULTIPLES)[number];

export interface SeriesSpec {
  key: string;
  label: string;
  color: string; // CSS custom property of the role
}

/** Models compared for one multiple. Colour follows the model family, in a fixed order. */
export function seriesForMultiple(multiple: Multiple): SeriesSpec[] {
  return [
    { key: `ML_ln${multiple}`, label: "GBM", color: "var(--series-1)" },
    { key: `BL_${multiple}`, label: "BL (regression)", color: "var(--series-2)" },
    { key: `IND_${multiple}`, label: "IND (industry mean)", color: "var(--series-3)" },
    { key: `HP_${multiple}`, label: "HP (text peers)", color: "var(--series-4)" },
  ];
}

export const TARGET_LABEL: Record<string, string> = {
  lnm2b: "Market-to-book",
  lnv2a: "Enterprise value to assets",
  lnv2s: "Enterprise value to sales",
};
