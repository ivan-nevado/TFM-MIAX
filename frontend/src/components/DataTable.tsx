import { formatCell } from "../features/firms/format";
import type { Cell, TableResponse } from "../types/api";

interface Props {
  table: TableResponse;
  maxRows?: number;
}

const isNumber = (v: Cell) => typeof v === "number";

/** Renders any replication table. Numeric columns are right-aligned with tabular figures. */
export function DataTable({ table, maxRows = 400 }: Props) {
  const rows = table.rows.slice(0, maxRows);
  const numeric = new Set(table.columns.filter((c) => rows.some((r) => isNumber(r[c])) && rows.every((r) => r[c] === null || isNumber(r[c]))));
  const longText = new Set(table.columns.filter((c) => rows.some((r) => typeof r[c] === "string" && (r[c] as string).length > 60)));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {table.columns.map((c) => (
              <th key={c} className={numeric.has(c) ? "num" : undefined}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {table.columns.map((c) => (
                <td key={c} className={numeric.has(c) ? "num" : longText.has(c) ? "wrap" : undefined}>
                  {formatCell(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.rows.length > maxRows && <p className="muted small">Showing the first {maxRows} of {table.rows.length} rows.</p>}
    </div>
  );
}
