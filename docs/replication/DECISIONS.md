# Decisions

Every choice made where the paper is silent or the data force one. Each has a config key (in
`ml/configs/replication.yaml`) or a code location, so it can be changed and rerun. None was chosen
by looking at model errors: data-definition choices used only the coverage comparison with Table
IA.4, and modelling defaults were fixed before any model was trained.

## A. Things settled by the paper's online appendix

The appendix (`gl-online-appendix.pdf`, public, downloaded by `scripts/paper_targets.py appendix`)
resolved points the master prompt marked "paper silent":

| Point | Appendix | What the code does |
|---|---|---|
| BL variables (Table IA.3) | indm2b, indv2a, indv2s, adjopmad, negadjopmad (adjopmad times an indicator for negative adjopmad), adjsalesgrowth, d2e = (dltt + dlc)/book, pretret_noa_wr, roe_wr, rd_sale_wr | Exactly these ten (`models/benchmarks/bl.py`) |
| RRV leverage (Table IA.1) | LEV1 = (dltt+dlc)/at, LEV2 = 1 − book/at, LEV3 = market leverage | LEV1 and LEV2. LEV3 contains the market value being predicted and is excluded as leakage (see DEVIATIONS) |
| BG items (Table IA.2) | 28 Compustat items | The 16 we have (`benchmarks.bg_items`); labelled approximate |
| 80:20 split (IA.6.3) | No validation set; number of trees fixed at 20 | `gbm.fixed_rounds_80_20: 20` (the prompt's fallback, median `best_iteration`, is not used) |
| CART (IA.6.2) | sklearn `DecisionTreeRegressor`, median imputation, depth 3–6 chosen on the validation set | Same (`benchmarks.cart_depths`) |
| Common sample (IA.5) | All models except HP | Same; a second table adds HP when available |

## B. Data ingestion (LSEG)

| # | Decision | Reason |
|---|---|---|
| B1 | Rows with every accounting item blank are dropped (607,063 of 943,690 quarterly rows; 142,450 of 237,957 annual) | They are padding of the download; what remains is exactly the 336,627 / 95,507 rows the team reported |
| B2 | Fundamentals divided by 1e6 (USD → USD millions). Delisted-firm market data: `mv` / 1e6 and `ret` / 100; listed-firm market data used as is | Verified on the files: listed `mv` median 607 (millions), `|ret|` median 0.064 (decimal); delisted `mv` median 6.2e8 (USD), `|ret|` median 7.0 (percent). `schemas.validate` fails if units are off |
| B3 | Delisted-firm dates moved to the calendar month end | They are last-trading-day dates |
| B4 | No sign changes | capx, dv, cogs, xsga, xint, dp are already positive (negative share below 1%) |
| B5 | Flows are single-quarter amounts and are summed over four quarters | Checked: the four-quarter sum equals the annual statement (median ratio 1.000 for sale, ib, ocf, capx, dp, dv) |
| B6 | `mii` = net income before minus after minority interest; `oibdp` := LSEG reported EBITDA | LSEG has no direct items; reported EBITDA equals ebit + dp (median ratio 1.00) |
| B7 | Universe: ISIN starts with `US` (else a CUSIP starting with a digit), not a REIT (TRBC 601020 or "REIT" in the name), RIC suffix `.N`, `.A`, `.OQ`/`.O` | Proxy for CRSP shrcd 10/11 and exchcd 1/2/3. NYSE = suffix `.N` |
| B8 | Financial firm (for `cash2assets_an`): SIC 6000–6999; where SIC is missing, TRBC sector Financials or Real Estate | SIC exists for 64% of firms |
| B9 | Monthly returns above +300% are set to missing (219 values) | Data errors (the team flagged them); affects beta and portfolio returns only. `panel.max_monthly_return` |
| B10 | `VTRS.OQ` (mixes two companies' history) is kept | One firm; `panel.exclude_ids` can drop it |

## C. Panel construction

| # | Decision | Reason |
|---|---|---|
| C1 | A fiscal period belongs to the month of (date − 10 days) | 52/53-week calendars end a few days around month end (2026-09-25, 2026-10-04 are both "September") |
| C2 | Quarter t−k is the record 3k months earlier, ±1 month; all four quarters are required for an annualised flow | Paper: "aggregating the values from quarter t−3 to quarter t". A missing quarter gives a missing flow |
| C3 | `l_x` = the record 12 months earlier (±1 month); for flows, the annualised value of that record | Reading of the paper's prefixes (the paper does not define them) |
| C4 | Availability: period month + 3 (a March quarter is first used in June) | Hou, Xue & Zhang (2018), as the paper |
| C5 | A quarterly record is used for at most 12 months after its date; an annual one for 18 | Paper silent. Stops stale statements being carried forever |
| C6 | Annual statements only where no usable quarterly record exists (2.2% of sample firm-months, 65% in 1990) | LSEG quarterly data start in 1989, so 1990 lacks the eight quarters needed for lags |
| C7 | A record is usable if total assets and annualised sales are present | Minimum for the three targets |
| C8 | Book equity = (ceq + pstk, else at − lt − mib) + txditc − pstk, missing txditc/pstk as zero | Davis, Fama & French (2000) with the items available |
| C9 | `seq` = ceq + pstk; `ibcom` = ibadj (else ib − preferred dividends); `icapt` = ceq + dltt + pstk + mib with missing as zero | Compustat items we do not have |
| C10 | `ebitda` = ebit + dp, falling back to reported EBITDA where dp is missing | Brings `roa_wr` coverage from 91% to 100% (paper: 100%) |
| C11 | Size filter keeps firms at or above the monthly 10th percentile of assets, book equity and sales, computed on firm-months with valid targets | Paper: "below the 10th percentile" are dropped |
| C12 | CAPM beta: 60-month window, all 60 months required | Paper silent; gives 69% coverage (paper 73%). Sensitivity: 24, 36, 48 |

## D. LSEG → Compustat definition switches (`panel.definitions`)

Chosen with the Table IA.4 coverage comparison only.

| Switch | Value | Evidence |
|---|---|---|
| `zero_fill_items` | dltt, dlc, pstk, invt, dv | LSEG leaves these blank when the firm has none; Compustat stores zero. Coverage: `debt_at_wr` 100% (paper 94%; `dlc` is blank in 27% of LSEG quarters), `quick_ratio_wr` 80% (paper 81%; without 67%), `invt_act_wr` 80% (paper 80%) |
| `opcost_from_ebitda` | true | cogs + xsga := sale − ebitda. In Compustat cogs + xsga = sale − oibdp exactly. LSEG puts depreciation inside cogs/xsga (ebit = sale − cogs − xsga, median ratio 1.00) and has no cogs for financial firms. `opleverage_an` and `profitability_an` coverage 78% → 100% (paper 100%) |
| `dp_adjustment` | none | Whether depreciation sits in cogs or in xsga varies by firm, so gross-margin variables keep LSEG's cost of revenue |
| `ebit_constructed` | false | Reported ebit already equals sale − cogs − xsga |
| `xsga_add_xrd` | false | LSEG SG&A already contains R&D (same identity) |
| `lt_includes_mib` | false | (lt + ceq + pstk + mib) / at = 1.00 |
| `nopi_spi_from_pretax` | false | spi + nopi (missing as zero) reconciles with pi − ebit + xint (median ratio 1.00) |
| `efftax_positive_pi_only` | true | WRDS convention; paper coverage 78% equals the share of profitable firm-months (ours 86%) |
| `dpr_positive_income_only` | true | Same reasoning; paper coverage 77% (ours 84%) |

Zero-fills inside formulas (marked `# zero-fill` in `features/accounting.py`), where Table IA.4
shows ~100% coverage although a Compustat input is often missing: xint in `profitability_an`;
xint, spi, nopi in `ptpm_wr`; nopi in `roic_an`; d_invt in `pay_turn_wr`.

## E. Modelling defaults (paper silent)

| # | Decision | Config |
|---|---|---|
| E1 | Blocks: random permutation dealt round-robin within the month (sizes differ by at most one), seed = base_seed + yyyymm, base_seed 20230301; firms sorted by id first | `splits` |
| E2 | Validation block of test block k is (k + 1) mod 5 | `splits/blocks.py` |
| E3 | Early stopping on validation L2, patience 50, cap 10,000 trees, predict with `best_iteration` | `gbm` |
| E4 | `industry` is a LightGBM categorical feature; a robustness run treats it as numeric | `gbm.industry_categorical` |
| E5 | `deterministic=True`, one thread, LightGBM seed unique per (base seed, month, block) | `gbm.deterministic` |
| E6 | A month-block with fewer than 40 training rows is skipped (complete-case models in the early 1990s) | `gbm.min_train_rows` |
| E7 | IND: at least 5 fitting firms per industry; RRV: at least 30 per industry-month | `benchmarks` |
| E8 | R²oos uses the mean of the three training blocks for the GBM and of the four fitting blocks for benchmarks | `models/runner.py` |
| E9 | LASSO: industry dummies, inputs standardised on the training block, penalty chosen on the validation block from a 50-point path | `models/lasso.py` |
| E10 | Seeds: five base seeds for the three base models; all other models use the first seed | `splits.seeds` |

## F. Evaluation and explainability

| # | Decision |
|---|---|
| F1 | Portfolios: quintile = 1 + number of NYSE breakpoints strictly below the error; a month needs at least five NYSE firms |
| F2 | Equal-weighted portfolios drop firms below the NYSE 20th percentile of market value (Hou, Xue & Zhang convention) |
| F3 | Sharpe ratio = mean / sd × √12 of the long-short return; information ratio = alpha / sd(residual) × √12 |
| F4 | Category `rho`: ranks reversed for the variables listed in `features/registry.py::REVERSED` (leverage- and debt-type ratios in Financial soundness, Solvency and Capitalization). The paper names only `dltt_be_wr` as an example |
| F5 | Decades are 1990s, 2000s, 2010s; `chg` = 2010s − 1990s (paper: 2010s − 1980s) |
| F6 | Peer weights: leaves whose last split is on the categorical `industry` are divided by n + `cat_l2` (10) by LightGBM, not by n. The decomposition reads the divisor of every leaf from the model, so `vᵀK` equals the prediction exactly and the weights still sum to one |
| F7 | Peer weights are stored for the months in `explain.peer_months` (December 2019): the 20 largest and 10 smallest weights of every test firm, and the full vector of the focus firm |
