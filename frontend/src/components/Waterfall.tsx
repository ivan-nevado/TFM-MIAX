import { useState } from "react";
import type { ShapItem } from "../types/api";

interface Props {
  items: ShapItem[];
}

const W = 640;
const ROW = 30;
const LABEL_W = 170;
const VALUE_W = 56;

/** SHAP waterfall: training mean, signed contributions (blue up, red down), prediction. */
export function Waterfall({ items }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  let running = 0;
  const bars = items.map((item) => {
    let start = 0;
    let end = item.value;
    if (item.kind === "base") running = item.value;
    else if (item.kind === "contribution") {
      start = running;
      end = running + item.value;
      running = end;
    }
    return { item, start, end };
  });
  const lo = Math.min(0, ...bars.map((b) => Math.min(b.start, b.end)));
  const hi = Math.max(0, ...bars.map((b) => Math.max(b.start, b.end)));
  const span = hi - lo || 1;
  const x = (v: number) => LABEL_W + ((v - lo) / span) * (W - LABEL_W - VALUE_W);
  const height = bars.length * ROW + 8;

  const color = (item: ShapItem) =>
    item.kind === "contribution" ? (item.value >= 0 ? "var(--diverging-pos)" : "var(--diverging-neg)") : "var(--neutral-mark)";

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="SHAP contributions to the predicted log multiple">
        <line x1={x(0)} x2={x(0)} y1={0} y2={height} stroke="var(--axis)" strokeWidth={1} />
        {bars.map(({ item, start, end }, i) => {
          const left = Math.min(x(start), x(end));
          const width = Math.max(Math.abs(x(end) - x(start)), 1);
          const label = item.kind === "contribution" ? `${item.value >= 0 ? "+" : ""}${item.value.toFixed(3)}` : item.value.toFixed(3);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={i * ROW} width={W} height={ROW} fill="transparent" />
              <text x={LABEL_W - 8} y={i * ROW + 19} textAnchor="end">
                {item.label}
              </text>
              <rect x={left} y={i * ROW + 6} width={width} height={ROW - 12} rx={3} fill={color(item)} />
              <text x={Math.max(x(start), x(end)) + 6} y={i * ROW + 19}>
                {label}
              </text>
            </g>
          );
        })}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: LABEL_W, top: hover * ROW + 30 }}>
          <strong>{bars[hover].item.label}</strong>
          {bars[hover].item.category ? ` (${bars[hover].item.category})` : ""}
          <div>
            {bars[hover].item.kind === "contribution"
              ? `moves the prediction from ${bars[hover].start.toFixed(3)} to ${bars[hover].end.toFixed(3)}`
              : `log multiple ${bars[hover].item.value.toFixed(3)}`}
          </div>
        </div>
      )}
    </div>
  );
}
