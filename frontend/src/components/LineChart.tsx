import { useMemo, useRef, useState } from "react";

export interface Series {
  key: string;
  label: string;
  color: string;
  values: (number | null)[];
}

interface Props {
  dates: string[]; // ISO dates, one per point
  series: Series[];
  yLabel: string;
}

const W = 920;
const H = 320;
const M = { top: 12, right: 16, bottom: 28, left: 44 };

function niceMax(v: number): number {
  const step = Math.pow(10, Math.floor(Math.log10(v || 1)));
  return Math.ceil(v / (step / 2)) * (step / 2);
}

/** Time-series line chart on one y-axis, with a legend, a crosshair and a tooltip. */
export function LineChart({ dates, series, yLabel }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const shown = series.filter((s) => s.values.some((v) => v !== null));

  const { x, y, yTicks, xTicks } = useMemo(() => {
    const all = shown.flatMap((s) => s.values.filter((v): v is number => v !== null));
    const yMax = niceMax(Math.max(...all, 0.1));
    const xScale = (i: number) => M.left + (i / Math.max(dates.length - 1, 1)) * (W - M.left - M.right);
    const yScale = (v: number) => H - M.bottom - (v / yMax) * (H - M.top - M.bottom);
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);
    const years = dates.map((d, i) => ({ i, year: d.slice(0, 4), month: d.slice(5, 7) })).filter((d) => d.month === "01");
    const every = Math.max(1, Math.ceil(years.length / 8));
    return { x: xScale, y: yScale, yTicks: ticks, xTicks: years.filter((_, k) => k % every === 0) };
  }, [dates, shown]);

  const path = (values: (number | null)[]) => {
    let d = "";
    let pen = false;
    values.forEach((v, i) => {
      if (v === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - M.left) / (W - M.left - M.right)) * (dates.length - 1));
    setHover(Math.min(Math.max(i, 0), dates.length - 1));
  };

  if (shown.length === 0 || dates.length === 0) return <p className="notice">No data for this chart yet.</p>;

  return (
    <div className="chart">
      <div className="legend" aria-label="Legend">
        {shown.map((s) => (
          <span key={s.key}>
            <span className="legend-swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by month`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={M.left - 6} y={y(t) + 4} textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t.i} x={x(t.i)} y={H - 8} textAnchor="middle">
            {t.year}
          </text>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--axis)" strokeWidth={1} />
        {shown.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
        ))}
        {hover !== null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} stroke="var(--axis)" strokeWidth={1} />
            {shown.map((s) => {
              const v = s.values[hover];
              return v === null ? null : (
                <circle key={s.key} cx={x(hover)} cy={y(v)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
              );
            })}
          </g>
        )}
        <rect
          x={M.left}
          y={M.top}
          width={W - M.left - M.right}
          height={H - M.top - M.bottom}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${Math.min((x(hover) / W) * 100, 72)}%`, top: 36 }}>
          <strong>{dates[hover].slice(0, 7)}</strong>
          {shown.map((s) => (
            <div key={s.key}>
              <span className="legend-swatch" style={{ background: s.color }} />
              {s.label}: {s.values[hover] === null ? "n/a" : (s.values[hover] as number).toFixed(3)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
