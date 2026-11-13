interface Props {
  label: string;
  value: string;
  sub?: string;
}

/** A headline number with its label and one line of context. */
export function StatTile({ label, value, sub }: Props) {
  return (
    <div className="card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
