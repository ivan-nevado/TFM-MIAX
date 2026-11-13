// Number formatting shared by tables and cards.
import type { Cell } from "../../types/api";

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "";
  if (Number.isInteger(value) && Math.abs(value) >= 1000) return value.toLocaleString("en-US");
  if (Math.abs(value) >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}

export function formatCell(value: Cell): string {
  if (value === null) return "";
  if (typeof value === "number") return formatNumber(value, 3);
  return String(value);
}

export function formatPercent(value: number, digits = 1): string {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

export function formatMillions(value: number): string {
  if (Math.abs(value) >= 1000) return `$${(value / 1000).toFixed(1)}bn`;
  return `$${value.toFixed(0)}m`;
}
