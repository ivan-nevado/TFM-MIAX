# Deviations from the paper

Two things are kept apart throughout the project:

- **Method fidelity** is exact: every step is implemented as the paper describes and checked by
  deterministic tests (`ml/tests`).
- **Numerical agreement** is measured, reported and never forced. Our numbers differ from the
  paper's because the sample differs. Measured gaps are in `REPLICATION_REPORT.md`; this file
  lists the causes.

## 1. Data

| Difference | Measured | Expected effect |
|---|---|---|
| Source: LSEG Workspace, not CRSP/Compustat | — | Item definitions differ (below) |
| Period 1990-01 to 2019-12 (360 months), paper 1980–2019 (480) | — | No 1980s column in Tables 8 and 9; totals compared per month, never raw |
| Survivorship: almost only firms listed today; the ~125 "delisted" firms with data left the market in 2024–2026 | 2,220 firms in the sample (paper 16,201); 1,170 firm-months per month (paper 3,775); 514 per month in 1990, 1,956 in 2019 | Fewer training rows per model (about 700 vs 2,265) raise errors, most in the 1990s; a sample tilted to large successful firms lowers them |
| Higher multiples | median lnm2b 0.65 / lnv2a 0.08 / lnv2s 0.63 (paper 0.48 / −0.03 / 0.35) | Consequence of the survivor sample |
| Universe proxy from ISIN/CUSIP, TRBC and RIC suffix, not shrcd/exchcd | 547,642 universe firm-months, 2,716 firms | Exchange is today's, not historical: NYSE breakpoints are approximate |
| Annual statements where no quarterly record is usable | 2.2% of sample firm-months (65% in 1990) | Annual data are older on average in 1990 |
| Restated fundamentals | — | Some look-ahead, as in Compustat quarterly |
| SIC is the current code and exists for 64% of firms | `industry` present in 73% of firm-months (paper 99%) | Industry less informative; IND, BL and RRV value fewer firms |
| Deferred taxes present in 44% of quarters | — | Book equity slightly understated where missing |
| Market value is company-level capitalisation | — | Slightly above CRSP's per-security value for dual-class firms |
| No cost of revenue for financial firms | `gpm_wr`, `gprof_wr` and their duplicates 78% coverage (paper 100%) | Gross-margin variables missing for banks and insurers |
| Receivables, inventories and current items sparse for financials | e.g. `debt2tang_an` 75% (paper 94%), `rect_turn_wr` 75% (91%) | More missing inputs; the GBM handles them |
| The 52 `_wr` ratios are computed by us, not taken from the WRDS suite | Four exact duplicate pairs | SHAP splits arbitrarily inside a pair; pairs are also reported summed |
| No delisting returns | — | Portfolio returns of firms that delist are missing in the delisting month |

Coverage, mean and median of every input next to the paper's Table IA.4 are in
`reports/replication/tables/ia4_summary_statistics.csv`.

### Sample size against the team's earlier panel builder

The earlier script (not in this repository) reported 547,480 → 472,843 → 402,906 firm-months.
This implementation gives 547,642 → 494,076 → 421,392 with the same 2,716 universe firms. The
4.5% more firm-months with valid targets come from documented choices that the earlier counts
cannot be checked against: records are used for up to 12 months (C5), the annual fallback (C6)
and the fiscal-month alignment (C1). Target medians (0.65, 0.08, 0.63 vs 0.64, 0.07, 0.66),
beta coverage (69%) and industry coverage (73% vs 74%) agree.

## 2. Method

| Difference | Reason | Effect |
|---|---|---|
| RRV uses two leverage measures, not three | The paper's LEV3 is market leverage, built from the market value being predicted: using it leaks the target | Our RRV should fit ln M worse than the paper's (R²oos(ŷ) 0.83) and is a cleaner benchmark |
| BG uses 16 of the 28 items | The rest are not in the download (teq, pstkr, ao, lo, aco, lco, ni, xido, ibcom, do; seq and icapt are linear combinations of items included) | Approximate; labelled so everywhere |
| HP not run | TNIC is keyed by gvkey (see BLOCKERS) | Table 2 has no HP rows |
| 13-factor model not run | The seven anomaly-cluster factors are not public (see BLOCKERS) | Table 3 has the mean and FF6 panels only |
| Complete-case models skip small months | With all 97 inputs required, some early months have fewer than 40 training firms | LASSO and GBM "nm" rows cover fewer months |
| LightGBM 3.2.1 as in the paper | Pinned in `ml/pyproject.toml` with NumPy < 2 | None |

## 3. "Paper silent" defaults

Listed with their config keys in `DECISIONS.md` section E (blocks and seeds, validation block,
early stopping, categorical industry, minimum sizes, R²oos training mean, LASSO penalty) and
section F (portfolio conventions, category orientation, decades, peer-weight divisors). The seed
spread of every Table 2 metric is reported next to the headline numbers.

## 4. What the sensitivity runs measure

| Limitation | Run | Table |
|---|---|---|
| Thin, biased early sample | Metrics by decade and by year | `table2_by_decade`, `table2_by_year` |
| Delisted firms | Listed-today firms only | `sensitivity_runs`, `sensitivity_listed_subsample` |
| Industry coding | Categorical against numeric | `sensitivity_runs` |
| Beta history | Minimum of 24, 36, 48, 60 months | `sensitivity_runs` |
| Block assignment | Five seeds | `table2_seed_spread` |
| Definition switches | Rebuild the panel with another `panel.definitions` and compare `ia4_summary_statistics` | See DECISIONS section D |
