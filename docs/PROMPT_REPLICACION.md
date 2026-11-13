# MASTER PROMPT — Replicate Geertsema & Lu (2023) and set up the thesis project

You are the engineer in charge of this repository. Read this whole document before touching anything. It tells you what the project is, what already exists, exactly what the paper does, the numbers you have to reproduce, how the repository must be organised, and when you are done.

---

## 0. Mission and operating rules

**Mission.** Replicate, end to end and with the paper's own methods, *Geertsema, P. & Lu, H. (2023), "Relative Valuation with Machine Learning", Journal of Accounting Research 61(1), 329–376*: same sample rules, same 97 input variables, same three targets, same LightGBM set-up, same benchmarks, same metrics, same portfolio tests, same SHAP analysis and same peer-weight decomposition. Then leave the project organised as a maintainable monorepo (ML package, backend, frontend, Terraform for AWS).

**The replication is the blocking priority.** Everything else in the thesis (LLM qualitative scores, other models, use cases, the final UI) comes later and is out of scope here, except for the scaffolding described in section 9.

**How to work**

1. Work autonomously from start to finish. Do not stop to ask for confirmation. When the paper is silent or the data forces a choice, use the default given in this document; if there is none, pick the option closest to the paper, write it in `docs/replication/DECISIONS.md` and keep going.
2. You are finished only when every item of the Definition of Done (section 11) is met. If one item is truly impossible (missing external data, for example), write it in `docs/replication/BLOCKERS.md` with the evidence and what would unblock it, and continue with everything else. A blocker never stops the rest of the work.
3. Keep `docs/replication/PROGRESS.md` up to date (what is done, what is running, what is next). The task is long: this file is how you resume after a context reset. Read it first whenever you start or resume.
4. Never fabricate, copy or hard-code a result. Every number in the reports must come from code in the repository that anyone can re-run. Never tune a method, a seed, a filter or a hyperparameter to make a number look closer to the paper: the method is fixed by the paper, the numbers are whatever the method gives on our data.
5. Never let the test block influence anything: not early stopping, not model selection, not filters, not imputation statistics.
6. The data are licensed (LSEG/Refinitiv). Never commit them to git, never upload them anywhere, never paste rows into external services. Treat everything under `DESCARGAS/`, `datos_lseg/` and `lseg_datos_mercado.zip` as read-only originals.
7. Do not create or change real cloud resources. Write the Terraform, run `terraform fmt` and `terraform validate`, and stop there. No `terraform apply`, no AWS credentials, no spending.
8. Initialise git if the folder is not a repository, add a `.gitignore` that excludes all data and outputs, and commit locally at each milestone. Do not push.
9. Write code and documentation in English. Existing Spanish files stay as they are.

---

## 1. The thesis in one paragraph (context only)

Master's thesis in Artificial Intelligence. Goal: a machine-learning model that determines a company's implied valuation multiple from its financial statements, its sector and qualitative scores produced by an LLM from the business description; then three use cases (valuing private companies, SHAP explanations of over/undervaluation, what-if scenarios such as "sales drop 20% with constant costs"), served through a cloud architecture with an interactive UI. The first step, and the only one that matters now, is the replication of Geertsema & Lu (2023). The later steps will reuse the panel, the trained models, the SHAP values and the peer weights you produce, so build them as reusable components and not as one-off scripts.

Files to read first:

| File | What it is |
|---|---|
| `J of Accounting Research - 2022 - GEERTSEMA - Relative Valuation with Machine Learning.pdf` | The paper (48 pages). The final authority on method. Table A.1 (pp. 367–369) has the 97 formulas |
| `Replica_paper1_y_extensiones_revisado_v3.docx` | The team's plan, in Spanish. Sections 2–11 and annexes A and C describe the replication; sections 12–14 and annex B are later extensions (out of scope) |
| `construir_97_variables.py` | Working panel builder (see section 2.3). Start from it |
| `CAMBIOS_v3.md`, `CAMBIOS_v4.md` | How the LSEG data were downloaded and how they differ from the paper's data |

The paper's online appendix (tables IA.1–IA.8, figures IA.1–IA.5) is not in the repository: `https://www.chicagobooth.edu/jar-online-supplements`. The authors' code for peer weights is at `https://github.com/pgeertsema/peerweights`, and a later related repository is `https://github.com/pgeertsema/HBR_peers`. If you have network access, fetch them and use them to settle details this document marks as "paper silent". If you do not, use the defaults given here.

---

## 2. What already exists

### 2.1 Honest starting point: the data are not the paper's data

The paper uses CRSP (monthly) and Compustat (quarterly) through WRDS, 1980–2019. We do not have WRDS. Our data come from LSEG Workspace (Refinitiv). This has consequences you must carry through the whole project:

| Difference | Effect |
|---|---|
| Period is **January 1990 – December 2019** (360 valuation months), not 1980–2019 (480) | LSEG has no quarterly fundamentals before 1989. All paper totals are compared per month or per year, never as raw totals |
| **Survivorship bias.** 4,902 currently listed firms and only 139 delisted ones | After filters we have about 1,119 firms per month on average (510 in 1990, 1,915 in 2019) against about 3,775 in the paper. Firms that died before today are almost all missing. Valuation multiples are higher than the paper's (median ln m2b 0.64 vs 0.48) |
| Accounting items are LSEG's standardised items, not Compustat's | Levels of some margins and turnover ratios differ. LSEG's cost of revenue and SG&A already include depreciation, so `ebit ≈ sale − cogs − xsga` |
| The 52 `_wr` ratios are computed by us from Table A.1, not taken from the WRDS Financial Ratios Suite | Four pairs of variables are exact duplicates (section 3.3) |
| Identifier is the RIC (`id`). No `permno`, `gvkey`, `shrcd`, `exchcd` | US incorporation is inferred from ISIN/CUSIP; REITs from TRBC code 601020 or "REIT" in the name; exchange from the RIC suffix (`.N` NYSE, `.A` NYSE American, `.OQ`/`.O` NASDAQ), which is today's exchange, not the historical one |
| SIC code is the current one and exists for 64% of firms | `industry` is present in 74% of firm-months (99% in the paper) |
| Market value is the company's total market capitalisation | Slightly larger than CRSP's per-security value for dual-class firms |
| Fundamentals are restated values (`Rsdt`) | Some look-ahead, as in Compustat quarterly |
| Deferred taxes (`txditc`) exist in 44% of quarters | Book equity is slightly understated where missing |

So there are two different things, and you must keep them apart in code and in every report:

- **Method fidelity (mandatory, exact).** Every step is implemented exactly as the paper describes. This is checked with deterministic tests (section 10).
- **Numerical agreement (measured, reported, never forced).** Our numbers will not equal the paper's, because the sample differs. You reproduce every table and figure, put ours next to the paper's, and explain each gap in `docs/replication/DEVIATIONS.md`.

Design the data layer with a source adapter (`lseg` today). If CRSP/Compustat files are ever placed in `data/raw/wrds/`, a `wrds` adapter must be the only thing needed to rerun the identical pipeline on the paper's own data and 1980–2019. Write the adapter interface and a documented stub for `wrds`.

### 2.2 Data files (read-only originals)

`datos_lseg/` holds the merged download. Sizes are large: never load a whole file just to look at it; read a few rows.

| File | Rows | Content and units |
|---|---|---|
| `lseg_fundamentales.csv` | 336,627 | Quarterly statements. `id` (RIC), `datadate` (fiscal period end), 30 Compustat-named items: `at act lct lt che invt rect ap ppent dltt dlc pstk ceq txditc mib sale cogs xsga xrd xint ebit dp pi txt ib prefdiv ibadj ocf capx dv`, plus `mii`, `nopi`, `spi`, `ebitda_lseg`, and `origen` (`final` = listed today, `desl` = delisted). **US dollars, not millions.** Income and cash-flow items are single-quarter values, not year-to-date |
| `lseg_fundamentales_anuales.csv` | 95,507 | Same columns, annual statements. Used only where a firm has no usable quarterly record (start-up lags of 1990–1991 and firms that only report annually; 1.2% of firm-months) |
| `lseg_mercado.csv` | 993,893 | `id`, `date` (month end), `ret` (monthly total return, decimal), `mv` (market capitalisation, **USD millions**), `origen`. 1980–2025 |
| `lseg_estaticos.csv` | 5,041 | `id`, `nombre`, `ticker`, `trbc_sector`, `trbc_industria`, `isin`, `cusip`, `sic`, `origen` |
| `F-F_Research_Data_Factors_CSV.zip`, `Siccodes49.zip` | | Ken French library: monthly Mkt-RF/SMB/HML/RF and the SIC ranges of the 49 industries |
| `lseg_prueba5_*.csv`, `lseg_campos_extra.csv`, `lseg_catalogo_campos.csv` | | Definition checks of LSEG fields against Compustat |
| `universo*.csv`, `lseg_universo_historico_todo.csv` | | Lists of RICs requested |

`DESCARGAS/` (raw zips) and `lseg_datos_mercado.zip` are the immutable sources of `datos_lseg/`. `unir_lotes_lseg.py` rebuilds `datos_lseg/` from them.

You still need, from the Ken French data library (`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/<file>`): `F-F_Research_Data_5_Factors_2x3_CSV.zip` and `F-F_Momentum_Factor_CSV.zip` (for the six-factor model) and `Siccodes12.zip` (Fama–French 12 industries). Download them into `data/external/` and record the download date.

Known data-quality points to guard with tests: about 219 monthly returns above 300% in `lseg_mercado.csv`; some RICs with no market value; one RIC (`VTRS.OQ`) that mixes two companies' history; `±inf` after divisions.

### 2.3 Existing code

- **`construir_97_variables.py`** builds the firm-month panel: reads the four CSVs, converts units to USD millions, fixes signs, sums four quarters, applies the three-month availability lag, computes book equity, the 95 accounting variables with the paper's formulas and missing-value semantics, CAPMbeta, the FF49 industry, the three targets and the Table 1 filters. It has been checked against hand calculations on synthetic data and runs on the real data. Run: `python construir_97_variables.py --datos datos_lseg --salida <dir>`. Output: `panel_97` (parquet, or csv.gz without pyarrow) with `id, date, datadate, fuente, origen`, the 97 variables, `lnm2b, lnv2a, lnv2s`, `M, B, A, S, D, V` (USD millions) and `ret`; plus `control_*.csv` files comparing coverage and sample size with the paper.
- **`unir_lotes_lseg.py`** merges the raw download zips into `datos_lseg/`.
- **`descarga_lseg_v4.ipynb`** is the download notebook (it only runs inside LSEG Workspace; you cannot run it).
- `panel_prueba/` is a trial output of the panel builder and `_revision/` is scratch: both are disposable. Regenerate, do not depend on them.

Current output of the panel builder on the real data (your refactor must reproduce these exactly before you change anything):

| Step (Table 1) | Firm-months | Firms | Per month | Paper per month |
|---|---|---|---|---|
| US common stocks on NYSE/AMEX/NASDAQ | 547,480 | 2,716 | 1,521 | 4,921 |
| Positive, non-missing targets | 472,843 | 2,523 | 1,313 | 4,505 |
| Assets, book equity and sales above the monthly 10th percentile | 402,906 | 2,223 | 1,119 | 3,775 |

Refactor this script into the package (section 9) without changing its results. Keep the original under `legacy/`. Port its checks to `pytest`.

---

## 3. The paper: sample, targets and inputs

### 3.1 Sample and timing

- Universe: US common stocks (CRSP `shrcd` 10 or 11) on NYSE, NASDAQ or AMEX (`exchcd` 1, 2, 3) with non-missing market equity. Our proxy: US-incorporated (ISIN prefix `US`, else CUSIP starting with a digit), not a REIT.
- Period: valuation months January 1990 to December 2019 inclusive (paper: 1980–2019).
- Filter 1: all three targets non-missing and positive (so `M > 0`, `B > 0`, `A > 0`, `S > 0`).
- Filter 2: drop firm-months with total assets, book equity or sales below that month's 10th percentile. The filter uses accounting size, not market value, to avoid leaking the target. It applies only to the sample the models are trained and evaluated on.
- No winsorising, no trimming, no imputation, and financial and regulated firms are kept.
- "Month" always means the valuation month. Accounting data become available three months after the balance-sheet date (Hou, Xue & Zhang 2018): a quarter ending 31 March is first used in the June cross-section.
- Income-statement and cash-flow items are annualised by summing quarters t−3 to t. Balance-sheet items are the latest quarter. Prefix `l_` is the same item one year (four quarters) earlier; `d_x = x − l_x`.

Paper's Table 1 (1980–2019): 2,361,860 firm-months and 19,697 firms; then 2,162,501 and 19,041; then 1,811,786 and 16,201 (the text says 1,811,785; unique firms by `gvkey`).

### 3.2 Targets and the conversion to equity value

| Symbol | Definition |
|---|---|
| `M` | Market value of equity at month end, USD millions (CRSP: `abs(altprc) × shrout`). Ours: `mv` |
| `B` | Book equity as in Davis, Fama & French (2000) |
| `A` | Total assets, latest available quarter |
| `S` | Sales, sum of the last four quarters |
| `D` | Book value of debt: `rowtotal(dltt, dlc, pstk)`, missing treated as zero. Cash is **not** subtracted |
| `V` | Enterprise value `M + D` |
| `m2b`, `v2a`, `v2s` | `M/B`, `(M+D)/A`, `(M+D)/S` |
| Targets | `lnm2b = ln(m2b)`, `lnv2a = ln(v2a)`, `lnv2s = ln(v2s)` |

One model per target. The model predicts `ŷ` (a log multiple). Predicted equity value, with no retransformation correction:

- `ML_lnm2b`: `M̂ = exp(ŷ) × B`
- `ML_lnv2a`: `M̂ = exp(ŷ) × A − D`
- `ML_lnv2s`: `M̂ = exp(ŷ) × S − D`

Models that predict a multiple in levels use `M̂ = m̂ × B`, `m̂ × A − D`, `m̂ × S − D`. A model that predicts `ln M` uses `M̂ = exp(ŷ)`; one that predicts `M` uses it directly. `M̂` can be negative for heavily indebted firms: keep those observations, count them and report the count.

Valuation error: **`ε = M̂ / M − 1`**. Negative `ε` means the model values the firm below the market (the stock is expensive according to the model); positive means cheap.

Reference (paper, Table IA.4): median `lnm2b` 0.48, `lnv2a` −0.03, `lnv2s` 0.35; means 0.53, −0.02, 0.38. Ours: medians 0.64, 0.07, 0.66.

### 3.3 The 97 input variables

Only fundamental (non-market) information, with one exception, the CAPM beta. Composition: 52 WRDS-ratio-suite ratios (`_wr`), 39 accounting-anomaly variables (`_an`), 4 size items (`assets`, `book`, `sale_ac`, `debt_book_value`), `CAPMbeta`, and `industry` (Fama–French 49, one column).

Formulas: Table A.1 of the paper, already implemented in `construir_97_variables.py`. If the script and the paper disagree, the paper wins and you record the fix. Conventions that matter:

- `sum(a, b)` is the SAS function: it ignores missing arguments and is missing only if all are missing. `rowtotal(...)` is Stata's: missing counts as zero.
- Division by zero or by a missing value gives missing. Conditions in the table ("if be > 0", "if lct > 0", "for positive denominator only") give missing when not met.
- `rd_sale_wr = max(xrd/sale, 0)` with missing `xrd` replaced by zero. `cash2assets_an` is missing for financial firms (SIC 6000–6999).
- `ebitda = ebit + dp`. `icapt = dltt + pstk + ceq + mib`. `earnat = ppent + act`. `noa = ppent + act − lct`. `be` is Davis–Fama–French book equity.
- `CAPMbeta`: slope of the firm's monthly excess return on `MKTRF` over the previous 60 months (t−60 to t−1). The paper does not state a minimum number of months; we require 60, which gives 69% coverage (paper: 73%).
- `industry`: FF49 from the SIC code with the `Siccodes49` ranges; a SIC outside every range is industry 49 ("Other"); no SIC gives missing.
- Exact duplicate pairs in our data, because we compute both from the same formula: `gpm_wr`=`grossmargin_an`, `gprof_wr`=`grossprofit_an`, `opmbd_wr`=`ebitda2revenue_an`, `at_turn_wr`=`assetturnover_an`. Keep all 97 columns in the models, as the paper does. In SHAP tables also report each pair summed, because the split of importance inside a duplicate pair is arbitrary.

The 11 categories (needed for the SHAP tables):

| Category | n | Variables |
|---|---|---|
| Profitability | 22 | aftret_eq_wr, aftret_equity_wr, aftret_invcapx_wr, cfm_wr, ebitda2revenue_an, efftax_wr, gpm_wr, gprof_wr, grossmargin_an, grossprofit_an, marginch_an, npm_wr, opmad_wr, opmbd_wr, pretret_earnat_wr, pretret_noa_wr, profitability_an, ptpm_wr, roa_wr, roce_wr, roe_wr, roic_an |
| Growth | 17 | assetgrowth_an, bookequitygrowth_an, capexgrowth_an, chbe_an, chca_an, chceq_an, chcl_an, chfnl_an, chlt_an, chnccl_an, deprn_an, inventorychange_an, inventorygrowth_an, investment_an, pchdeprn_an, rd_sale_wr, salesgrowth_an |
| Financial soundness | 16 | Accrual_wr, cash_debt_wr, cash_lt_wr, curr_debt_wr, debt_ebitda_wr, dltt_be_wr, fcf_ocf_wr, int_debt_wr, int_totdebt_wr, invt_act_wr, lt_debt_wr, lt_ppent_wr, ocf_lct_wr, profit_lct_wr, rect_act_wr, short_debt_wr |
| Efficiency | 15 | assetturnover2_an, assetturnover_an, at_turn_wr, invturn_wr, opleverage_an, pay_turn_wr, pchgm2pchsale_an, pchsale2pchinvt_an, pchsale2pchrect_an, pchsale2pchxsga_an, pchsales2inv_an, rect_turn_wr, sales2cash_an, sales2inv_an, sales2rec_an |
| Liquidity | 8 | cash2assets_an, cash_conversion_wr, cash_ratio_wr, chcurrentratio_an, curr_ratio_wr, liquid2assets_an, pchquickratio_an, quick_ratio_wr |
| Solvency | 8 | cf2debt_an, de_ratio_wr, debt2tang_an, debt_assets_wr, debt_at_wr, debt_capital_wr, intcov_ratio_wr, intcov_wr |
| Capitalisation | 4 | capital_ratio_wr, debt_invcap_wr, equity_invcap_wr, totdebt_invcap_wr |
| Size | 4 | assets, book, debt_book_value, sale_ac |
| Industry | 1 | industry |
| Payout | 1 | dpr_wr |
| CAPMbeta | 1 | CAPMbeta |

Before training, reproduce the paper's summary-statistics check (Table IA.4): coverage, mean and median of every variable next to the paper's coverage. The panel builder already writes this (`control_variables.csv`).

---

## 4. The paper: models

### 4.1 Gradient boosting machine (the base models `ML_lnm2b`, `ML_lnv2a`, `ML_lnv2s`)

- Library: **LightGBM 3.2.1** (Microsoft). Try to pin that version in an isolated environment (with a NumPy it supports). If it cannot be installed, use the newest LightGBM, record the version, and state it as a deviation. All hyperparameters are **LightGBM defaults**: `objective=regression` (L2), `learning_rate=0.1`, `num_leaves=31`, `min_data_in_leaf=20`, everything else default. The only tuned quantity is the number of trees, chosen by early stopping on the validation set.
- Training is **cross-sectional, month by month**: each model sees one month only.
- Split, each month: observations are randomly allocated to five blocks. Each block is the test set once. Of the four remaining blocks, three are the training set and one is the validation set (60:20:20). So five models per month per target: `5 × 360 × 3 = 5,400` models (paper: `5 × 480 × 3 = 7,200`). Every firm-month gets exactly one out-of-sample prediction per target. **Every result in the paper is computed from these out-of-sample predictions.**
- Missing inputs are passed as `NaN`; LightGBM handles them. Replace `±inf` by `NaN`.
- Paper silent, defaults to use:
  - Block assignment: random permutation within the month, near-equal block sizes, seed `base_seed + yyyymm`, `base_seed = 20230301`. The same blocks are used by every model and benchmark.
  - Validation block for test block `k`: block `(k + 1) mod 5`.
  - Early stopping: validation L2, patience 50 rounds, cap 10,000 rounds; predict with `best_iteration`. Save the distribution of `best_iteration`.
  - `industry` is passed as a LightGBM categorical feature (the paper treats it as a categorical variable). Run one robustness pass with it as a plain numeric column and report the difference.
  - LightGBM's own randomness: `seed` fixed, `deterministic=True`, single thread for bit-reproducibility where needed.
- Seed sensitivity: repeat the three base models with 5 different `base_seed` values and report the spread of every Table 2 metric. The headline numbers are those of the first seed, with the spread next to them.

**Master predictions table.** Store one row per (model, id, date): `model, target, id, date, block, seed, y, yhat, M, Mhat, eps, n_train, train_mean_y, train_mean_M, best_iteration`. Every metric, figure, portfolio and comparison is computed from this table, so all analyses use exactly the same predictions. Benchmarks write to the same schema.

### 4.2 Benchmarks

All benchmarks use the same five blocks. They have no validation need, so they are fitted on the **four non-test blocks** and predict the test block. They require non-missing values for all their inputs, so they produce fewer predictions (paper: BL 1,335,453; IND 1,738,553; HP about 905,570; RRV 1,576,212; BG 1,466,704, against 1,811,785 for the GBM). Report results on each model's own valid predictions (Table 2) and on the common subsample where all models have a prediction (Table IA.5).

| Model | Target | Specification |
|---|---|---|
| `IND_m2b`, `IND_v2a`, `IND_v2s` (Liu, Nissim & Thomas 2002) | multiple, levels | Predicted multiple = **harmonic mean** `n / Σ(1/mᵢ)` of the multiples of the fitting-sample firms in the same FF49 industry and month. Paper silent on a minimum number of firms: use 5 |
| `BL_m2b`, `BL_v2a`, `BL_v2s` (Bhojraj & Lee 2002) | multiple, levels | Monthly cross-sectional OLS of the multiple on 10 variables (online appendix Table IA.3): the three industry harmonic means of m2b, v2a, v2s; industry-adjusted EBIT margin (`opmad_wr` minus the industry median) and the same variable multiplied by an indicator for negative values; industry-adjusted sales growth; leverage `(dltt + dlc) / book`; `pretret_noa_wr`; `roe_wr`; `rd_sale_wr`. Industry statistics are computed on the fitting sample only. Check the exact list against Table IA.3 if you can fetch the online appendix |
| `HP_m2b`, `HP_v2a`, `HP_v2s` (Hoberg & Phillips 2010, 2016) | multiple, levels | For each firm, the 60 firms with the highest text-similarity score (TNIC data, `https://hobergphillips.tuck.dartmouth.edu/industryclass.htm`) based on the prior year's 10-K; prediction = mean of those peers' current multiples; at least 5 non-missing peers. TNIC is keyed by `gvkey`, which we do not have. Build a `gvkey`↔RIC link only if a legitimate free mapping exists (for example through CUSIP in the TNIC files); otherwise record HP as a blocker |
| `RRV` (Rhodes-Kropf, Robinson & Viswanathan 2005) | `ln M` | Industry-month cross-sectional OLS of `ln M` on log book equity, log absolute net income, a loss indicator and leverage. The team's plan says online appendix Table IA.1 lists three leverage measures: take the exact list from there if you can fetch it. Otherwise use `ln M = a₀ + a₁ ln B + a₂ ln|ib| + a₃ I(ib<0)·ln|ib| + a₄ LEV` with book leverage `LEV = 1 − B/A`, and record it. Never use a leverage measure built with `M`: it would leak the target. Minimum 30 firms per industry-month regression (paper silent) |
| `BG` (Bartram & Grinblatt 2018) | `M`, levels | Monthly cross-sectional OLS of market value on 28 accounting items (online appendix Table IA.2). We have 33 items, not BG's exact 28: use the available items that match the list, record which are missing, and label the result as approximate |

Extra models used in the paper's robustness and reconciliation sections:

- **OLS on logs** with the 10 BL variables (Table 4, row 2).
- **GBM with the 10 BL variables** (Table 4, row 3) and **GBM with all 97** on the BL sample (row 4).
- **LASSO** on the 97 variables (industry as dummies, inputs standardised on the fitting sample), only on observations with all variables present ("nm" = non-missing; 273,225 in the paper), with the GBM rerun on those same observations (Table 4, rows 5–6). Penalty chosen on the validation block (paper silent).
- **CART** (online appendix IA.6.2): a single regression tree, depth chosen on the validation block. Paper result: median absolute error 0.35 / 0.39 / 0.44 for m2b / v2a / v2s, better than the traditional models and worse than the GBM on every metric.
- **Alternative GBM targets** (IA.6.1): equity value, log equity value, and the multiples in levels. Paper result: log multiples are best.
- **Alternative split 80:20** (IA.6.3, Table IA.6): four blocks train, one test, no validation set. Paper silent on the number of trees: use the median `best_iteration` of the base run for that month and target. Paper result: marginally weaker than the base models, still far better than the traditional models.

---

## 5. The paper: evaluation

### 5.1 Metrics (Table 2)

For each model, pooled over all its out-of-sample predictions:

- `N`; `med(|ε|)` (the headline metric); `avg(|ε|)`; `med(ε)`; `avg(ε)`; `sd(ε)`.
- `rmse(x̂) = sqrt( (1/N) Σ (x − x̂)² )`. `rmse(ŷ)` is on the model's own target; `rmse(M̂)` is on equity value in USD millions.
- Out-of-sample R²: `R²oos(p̂) = 1 − Σ(p − p̂)² / Σ(p − p̄)²`, where `p̄` is **the mean of the training sample** of the model that produced the prediction (so each observation uses its own model's training mean). Report `R²oos(ŷ)` (model) and `R²oos(M̂)` (dollar).
- `ρ(M, M̂)`: Pearson correlation between actual and predicted equity value.

Dollar metrics are the ones comparable across models with different targets.

### 5.2 Over time and across firms (Figures 1–3)

- Figure 1: monthly `med(|ε|)` of each model. Panels A–C: GBM against BL, IND, HP for the same multiple. Panel D: the three GBMs against RRV and BG. Panel E: the three GBMs together. Paper: the GBM line is below every traditional model in **every month**; errors rise in 2000 and 2008; the advantage of `ML_lnm2b` over the other two GBMs narrows over time.
- Figure 2: each month sort firms into quintiles by market capitalisation, by market-to-book and by ROE; plot `med(|ε|)` per quintile, GBM against the traditional models.
- Figure 3: for the three GBMs, `med(|ε|)` (panel A) and `med(ε)` (panel B) by size, market-to-book and ROE quintile and by Fama–French 12 industry. Paper: small firms, extreme market-to-book and low ROE are harder to value; Energy, Healthcare and Telecom are harder, Money and Utilities easier; the GBM overvalues small and low market-to-book firms and undervalues large and high market-to-book firms; bias does not vary much across ROE quintiles or industries.

### 5.3 Valuation errors and future returns (Table 3, Table IA.7)

Purpose: if the model value is a good proxy for fundamental value, prices should converge to it, so this month's valuation error should predict next month's return.

- At the end of each month `t`, sort stocks into quintiles by `ε` using **NYSE breakpoints**. Quintile 1 = most expensive (most negative `ε`), quintile 5 = cheapest.
- Hold during month `t+1`. Value-weighted by month-`t` equity value. Long quintile 5, short quintile 1.
- Errors from valuation months January 1990–December 2019 give 359 monthly returns (paper: 479).
- Returns for `t+1` are used for every stock that has one, with no size filter at `t+1`.
- Report: mean monthly return; annualised Sharpe ratio; alpha and loadings on the Fama–French six-factor model (MKTRF, SMB, HML, RMW, CMA, UMD); adjusted R²; annualised information ratio. t-statistics with Newey–West standard errors, 3 lags; stars at 10%, 5%, 1% (two-tailed).
- The paper also uses a 13-factor model (FF6 plus seven anomaly-cluster factors from Geertsema & Lu 2020). Those seven factor series are not public: record as a blocker unless you find the authors' data.
- Equal-weighted version (Table IA.7): NYSE breakpoints and firms below the 20th percentile of market capitalisation each month excluded.
- Our limits: NYSE membership comes from the RIC suffix `.N` (current exchange); delisting returns are not available; with survivorship bias the result is a weaker test than the paper's. Say so in the report.

### 5.4 Sources of the improvement (Table 4)

Start from the BL model and change one thing at a time on the **same observations** (those with all 10 BL variables; 1,335,453 in the paper): (1) OLS, BL variables → multiple; (2) OLS, BL variables → log multiple; (3) GBM, BL variables → log multiple; (4) GBM, all 97 variables → log multiple. Then, on the complete-case subsample: (5) LASSO, 97 variables → log multiple; (6) GBM, 97 variables → log multiple. Paper's conclusion: the log transformation and the switch from OLS to GBM are the two big gains; more variables matter little except for v2s; the GBM beats LASSO, so its edge is non-linearities and interactions, not variable selection.

### 5.5 Variable importance with SHAP (Tables 7, 8, 9, IA.8; Figure 5)

- Exact TreeSHAP values for the out-of-sample (test) observations of each model (`pred_contrib=True` in LightGBM, or `shap.TreeExplainer`). For each observation, the raw SHAP values plus the base value sum to the prediction.
- **Scaled absolute SHAP**: for each observation take absolute values and scale them so the 97 sum to 100%. All SHAP results in the paper are averages of these scaled values, in percentage points.
- Variable importance = average over all observations (Table 7: top 10 per model, with cumulative SHAP). Category importance = sum over the variables of a category (Table 8), for the full sample and by decade (paper: 1980s, 1990s, 2000s, 2010s; ours: 1990s, 2000s, 2010s, and `Chg` = 2010s − 1990s).
- `rho` in Table 7: correlation between the rank percentile of the variable and the rank percentile of the predicted target (rank percentiles computed within month). In Table 8, `rho` uses the average rank percentile of the category's variables, re-oriented where needed so a higher rank means "more of the category label" (for example a lower debt-to-equity ratio ranks higher in Financial soundness).
- Table 9: for each model take the 10 most important variables of the first decade and of the last decade, then report the 10 with the largest absolute change.
- Figure 5: raw SHAP waterfall for one firm (Moderna, December 2019): training mean, top five contributions, "other", prediction.

Theory the importances are checked against (paper, section 5.1 and Table 6), from a Gordon growth model:

- `m2b = ROE · [1 − θe(1 − δ)] · (1 + ge)/(re − ge)`, and with DuPont `ROE = PM1 · ATR / (1 − LEV)`.
- `v2a = ROA · (1 − θv) · (1 + g)/(WACC − g) = PM2 · ATR · (1 − θv) · (1 + g)/(WACC − g)`.
- `v2s = PM2 · (1 − θv) · (1 + g)/(WACC − g) = (1 − oplev/ATR)(1 − t)(1 − θv)(1 + g)/(WACC − g) = (ROA/ATR)(1 − θv)(1 + g)/(WACC − g)`.

Expected signs: Profitability and Growth positive for all three; Efficiency positive for m2b and v2a and **negative for v2s**; CAPMbeta, Size and Payout negative; Financial soundness, Solvency, Liquidity and Capitalisation ambiguous. ROE-type ratios should lead m2b, ROA-type ratios v2a, and operating leverage, asset turnover and EBIT margin v2s.

### 5.6 Peer weights (section 4.5, Appendix A.3, Table 5, Figure 4)

Any GBM prediction can be written exactly as a weighted average of the training-sample target values. Notation: `N` training firms, `v` the `N×1` vector of their targets, `λ` the learning rate (0.1), `M` trees, `I` the identity, `1` an all-ones matrix, `⊘` element-wise division.

- Iteration 0: `P₀ = (1/N)·1_{N×N}`, `G₀ = P₀`.
- For each tree `i = 1…M`:
  1. `Wᵢ = Dᵢ ⊘ (1_{N×N} Dᵢ)`, where `Dᵢ[f,c] = 1` if training firms `f` and `c` fall in the same leaf of tree `i`. So `Wᵢ[f,c]` is one over the leaf size if they share a leaf, else 0.
  2. `Pᵢ = λ · Wᵢ · (I − Gᵢ₋₁)`.
  3. `Gᵢ = Gᵢ₋₁ + Pᵢ`.
- Out-of-sample weights for `S` test firms: `K = Σᵢ₌₀..M Pᵢᵀ (Lᵢ ⊘ (1_{N×N} Lᵢ))`, where `Lᵢ` is the `N×S` indicator that test firm `s` shares a leaf with training firm `f` in tree `i` (for `i = 0` all firms share one leaf). Predictions: `k = vᵀ K`.

Use only the trees up to `best_iteration`. The identity holds because with L2 loss and default settings (no regularisation, `boost_from_average`) each leaf value is the mean residual of its training observations.

Reproduce: Table 5 (the 10 most and 10 least comparable peers of Moderna, `MRNA.OQ`, in December 2019, with `lnm2b`, weight, cumulative weight, FF12 and FF49 industry, `rd_sale_wr`, `cash2assets_an`, `de_ratio_wr`, `roe_wr`) and Figure 4 (Moderna's peer weights sorted in descending order: a steep start, a long flat section near zero, a small negative tail). In the paper Moderna's actual `lnm2b` is 1.66 (ours: 1.65) and the predicted value 1.36; eight of the ten closest peers are in the FF "Drugs" industry.

---

## 6. Numbers to reproduce (paper, 1980–2019)

Transcribe these tables into machine-readable files under `docs/replication/paper_targets/` as your first task, check them against the PDF, and generate every comparison table from those files. Minus signs here are ASCII `-`.

### 6.1 Table 2 — model performance

```csv
model,N,med_abs_eps,avg_abs_eps,med_eps,avg_eps,sd_eps,rmse_y,rmse_M,r2oos_y,r2oos_M,rho
ML_lnm2b,1811785,0.29,0.48,-0.00,0.19,1.14,0.56,7289,0.54,0.78,0.90
ML_lnv2a,1811785,0.30,0.54,0.01,0.25,1.53,0.42,7395,0.75,0.77,0.89
ML_lnv2s,1811785,0.32,0.59,0.01,0.28,4.66,0.44,7537,0.80,0.76,0.89
BL_m2b,1335453,0.52,1.16,0.21,0.73,3.91,6.47,12437,0.12,0.38,0.79
BL_v2a,1335453,0.67,1.52,0.60,1.30,4.41,2.00,11414,0.20,0.48,0.86
BL_v2s,1335453,0.68,2.25,0.20,0.85,14.13,7.29,12245,0.21,0.40,0.83
IND_m2b,1738553,0.42,0.58,-0.20,0.02,1.32,6.58,12200,-0.03,0.38,0.64
IND_v2a,1738553,0.47,0.86,-0.15,0.12,6.06,2.13,13029,0.02,0.29,0.61
IND_v2s,1738553,0.64,1.29,-0.35,0.00,12.89,8.10,17078,-0.01,-0.22,0.38
HP_m2b,905568,0.40,0.61,-0.16,0.10,1.80,6.21,13079,-0.03,0.51,0.73
HP_v2a,905566,0.48,0.93,-0.17,0.04,7.74,1.84,16681,0.09,0.20,0.60
HP_v2s,905573,0.59,1.28,-0.22,0.17,9.55,9.64,30563,-1.57,-1.67,0.28
RRV,1576212,0.43,0.73,0.02,0.35,2.07,0.76,12050,0.83,0.41,0.65
BG,1466704,0.86,3.84,-0.17,-1.58,32.05,24186.12,24186,-1.35,-1.35,0.42
```

On the common subsample (Table IA.5) the paper's `med(|ε|)` is 0.31–0.32 for the GBMs, 0.45–0.63 for IND and 0.52–0.68 for BL.

### 6.2 Table 3 — value-weighted long–short portfolios (monthly, percentage points)

Column order: `ML_lnm2b, ML_lnv2a, ML_lnv2s, BL_m2b, BL_v2a, BL_v2s, IND_m2b, IND_v2a, IND_v2s, HP_m2b, HP_v2a, HP_v2s, RRV, BG`.

```csv
panel,row,ML_lnm2b,ML_lnv2a,ML_lnv2s,BL_m2b,BL_v2a,BL_v2s,IND_m2b,IND_v2a,IND_v2s,HP_m2b,HP_v2a,HP_v2s,RRV,BG
Mean,Const,0.37***,0.42***,0.35***,0.29*,0.32*,0.30**,0.18,0.16,0.31**,0.10,0.09,0.38**,0.33**,0.28**
Mean,Sharpe Ratio,0.44,0.54,0.44,0.31,0.28,0.36,0.18,0.18,0.33,0.10,0.10,0.40,0.37,0.35
FF6,MKTRF,-0.03,-0.01,0.05**,-0.05,-0.00,0.13***,0.01,0.07***,-0.02,-0.00,-0.01,-0.11***,0.10***,0.04
FF6,SMB,0.09**,0.04,0.05,0.15***,0.15***,0.06,0.13***,0.10***,0.14***,0.14***,0.16***,0.22***,0.47***,-0.20***
FF6,HML,0.32***,0.21***,0.29***,0.24***,0.58***,0.01,0.59***,0.48***,-0.14**,0.41***,0.19***,-0.02,0.40***,0.27***
FF6,RMW,0.07,0.01,0.07,-0.17***,-0.05,0.06,-0.23***,-0.01,0.58***,-0.18***,0.06,0.25***,-0.00,0.28***
FF6,CMA,-0.08,0.04,0.08,-0.08,-0.02,0.09,-0.07,0.13*,0.57***,0.18**,0.41***,0.60***,0.05,0.05
FF6,UMD,-0.30***,-0.27***,-0.27***,-0.30***,-0.42***,-0.12***,-0.30***,-0.26***,-0.11***,-0.27***,-0.24***,-0.16***,-0.21***,-0.16***
FF6,Const,0.46***,0.51***,0.35***,0.51***,0.43***,0.23*,0.30***,0.11,0.05,0.21,0.08,0.31**,0.23**,0.18
FF6,Adjusted R2,0.36,0.30,0.36,0.28,0.52,0.07,0.55,0.50,0.25,0.44,0.35,0.31,0.52,0.35
FF6,Information Ratio,0.71,0.80,0.55,0.66,0.58,0.31,0.48,0.17,0.07,0.31,0.12,0.43,0.39,0.29
FF6+7,Const,0.38***,0.46***,0.29**,0.42***,0.45***,0.21,0.28**,0.16,0.13,0.28**,0.25*,0.43***,0.26**,0.29**
FF6+7,Adjusted R2,0.42,0.36,0.43,0.31,0.53,0.08,0.57,0.51,0.28,0.47,0.39,0.34,0.55,0.36
FF6+7,Information Ratio,0.62,0.76,0.49,0.56,0.61,0.28,0.46,0.27,0.17,0.44,0.39,0.61,0.46,0.47
```

Pattern to look for: the three GBM strategies have positive, significant mean returns and alphas, positive HML and negative UMD loadings, and higher Sharpe ratios (0.44–0.54) than the traditional models (0.10–0.40).

### 6.3 Table 4 — sources of the performance gains

```csv
panel,model,N,med_abs_eps,avg_abs_eps,rmse_y,rmse_M,r2oos_y,r2oos_M,rho
m2b,OLS : BL vars -> m2b,1335453,0.52,1.16,6.47,12437,0.12,0.38,0.79
m2b,OLS : BL vars -> ln(m2b),1335453,0.40,0.67,0.71,8624,0.30,0.70,0.85
m2b,GBM : BL vars -> ln(m2b),1335453,0.33,0.56,0.63,7426,0.46,0.78,0.89
m2b,GBM : ML vars -> ln(m2b),1335453,0.31,0.50,0.58,7632,0.54,0.77,0.89
m2b,LASSO : ML vars (nm) -> ln(m2b),273225,0.30,0.45,0.51,14200,0.51,0.76,0.89
m2b,GBM : ML vars (nm) -> ln(m2b),273225,0.26,0.39,0.46,13897,0.63,0.78,0.90
v2a,OLS : BL vars -> v2a,1335453,0.67,1.52,2.00,11414,0.20,0.48,0.86
v2a,OLS : BL vars -> ln(v2a),1335453,0.56,1.29,0.54,9585,0.33,0.63,0.86
v2a,GBM : BL vars -> ln(v2a),1335453,0.34,0.61,0.48,6721,0.48,0.82,0.91
v2a,GBM : ML vars -> ln(v2a),1335453,0.31,0.54,0.44,7296,0.61,0.79,0.90
v2a,LASSO : ML vars (nm) -> ln(v2a),273225,0.40,0.81,0.39,13375,0.47,0.79,0.89
v2a,GBM : ML vars (nm) -> ln(v2a),273225,0.26,0.40,0.34,13179,0.67,0.80,0.91
v2s,OLS : BL vars -> v2s,1335453,0.68,2.25,7.29,12245,0.21,0.40,0.83
v2s,OLS : BL vars -> ln(v2s),1335453,0.54,1.22,0.73,12756,0.46,0.35,0.78
v2s,GBM : BL vars -> ln(v2s),1335453,0.39,0.72,0.54,7713,0.70,0.76,0.89
v2s,GBM : ML vars -> ln(v2s),1335453,0.32,0.56,0.45,7165,0.79,0.79,0.91
v2s,LASSO : ML vars (nm) -> ln(v2s),273225,0.32,0.51,0.41,15511,0.73,0.72,0.85
v2s,GBM : ML vars (nm) -> ln(v2s),273225,0.26,0.41,0.35,12934,0.84,0.81,0.92
```

### 6.4 Table 7 — ten most important variables per model (scaled absolute SHAP, %)

```csv
model,variable,category,shap,cshap,rho,expected_sign
ML_lnm2b,industry,Industry,8.9,8.9,-0.139,NA
ML_lnm2b,roe_wr,Profitability,7.7,16.6,0.536,+
ML_lnm2b,profitability_an,Profitability,6.0,22.6,0.490,+
ML_lnm2b,rd_sale_wr,Growth,4.9,27.5,0.269,+
ML_lnm2b,salesgrowth_an,Growth,3.8,31.3,0.368,+
ML_lnm2b,aftret_invcapx_wr,Profitability,3.7,34.9,0.554,+
ML_lnm2b,pretret_noa_wr,Profitability,3.0,38.0,0.493,+
ML_lnm2b,roic_an,Profitability,2.9,40.8,0.518,+
ML_lnm2b,roce_wr,Profitability,2.4,43.3,0.517,+
ML_lnm2b,CAPMbeta,CAPMbeta,2.1,45.3,0.170,-
ML_lnv2a,industry,Industry,7.7,7.7,-0.335,NA
ML_lnv2a,roic_an,Profitability,7.0,14.7,0.535,+
ML_lnv2a,chbe_an,Growth,4.3,19.0,0.491,+
ML_lnv2a,lt_ppent_wr,Financial soundness,4.1,23.1,-0.353,+/-
ML_lnv2a,debt_at_wr,Solvency,4.1,27.2,-0.121,+/-
ML_lnv2a,sales2rec_an,Efficiency,3.9,31.1,0.295,+
ML_lnv2a,rd_sale_wr,Growth,3.8,34.9,0.352,+
ML_lnv2a,pretret_earnat_wr,Profitability,3.3,38.2,0.494,+
ML_lnv2a,salesgrowth_an,Growth,3.2,41.4,0.319,+
ML_lnv2a,lt_debt_wr,Financial soundness,2.9,44.3,0.016,+/-
ML_lnv2s,opleverage_an,Efficiency,21.2,21.2,-0.819,-
ML_lnv2s,at_turn_wr,Efficiency,9.0,30.2,-0.800,-
ML_lnv2s,opmad_wr,Profitability,6.4,36.7,0.533,+
ML_lnv2s,industry,Industry,5.9,42.6,0.112,NA
ML_lnv2s,npm_wr,Profitability,3.4,46.0,0.455,+
ML_lnv2s,rd_sale_wr,Growth,2.5,48.5,0.072,+
ML_lnv2s,salesgrowth_an,Growth,2.4,50.8,0.191,+
ML_lnv2s,debt_at_wr,Solvency,2.1,53.0,-0.045,+/-
ML_lnv2s,gpm_wr,Profitability,2.1,55.1,0.539,+
ML_lnv2s,chbe_an,Growth,2.1,57.2,0.166,+
```

### 6.5 Table 8 — category importance (scaled absolute SHAP, %), full sample and by decade

```csv
model,category,shap,rho,expected_sign,d1980,d1990,d2000,d2010,chg
ML_lnm2b,Profitability,38.0,0.648,+,38.6,39.1,36.6,37.6,-1.1
ML_lnm2b,Growth,20.6,0.432,+,15.9,22.9,21.3,21.5,5.6
ML_lnm2b,Industry,8.9,-0.123,NA,11.1,8.7,8.3,7.4,-3.7
ML_lnm2b,Financial soundness,8.7,0.184,+/-,9.1,7.8,8.6,9.9,0.8
ML_lnm2b,Efficiency,6.0,0.228,+,6.4,5.6,6.1,6.0,-0.4
ML_lnm2b,Liquidity,5.7,0.198,+/-,5.2,4.4,7.0,6.4,1.2
ML_lnm2b,Solvency,4.6,0.334,+/-,4.1,3.7,5.4,5.7,1.6
ML_lnm2b,Size,3.7,0.024,-,4.4,4.2,3.3,2.3,-2.1
ML_lnm2b,CAPMbeta,2.1,0.142,-,3.4,1.8,1.8,1.3,-2.2
ML_lnm2b,Capitalization,1.1,0.119,+/-,1.2,1.1,1.1,1.1,-0.2
ML_lnm2b,Payout,0.6,-0.034,-,0.6,0.7,0.5,0.9,0.4
ML_lnv2a,Profitability,24.9,0.452,+,22.2,25.5,26.3,25.1,2.8
ML_lnv2a,Growth,17.4,0.379,+,14.9,19.6,17.3,16.5,1.6
ML_lnv2a,Financial soundness,16.6,0.246,+/-,22.9,16.9,12.4,14.3,-8.6
ML_lnv2a,Solvency,12.3,0.571,+/-,12.8,14.1,12.2,8.8,-4.1
ML_lnv2a,Efficiency,10.7,0.449,+,6.0,7.4,12.5,19.5,13.6
ML_lnv2a,Industry,7.7,-0.380,NA,8.8,7.1,8.5,6.3,-2.5
ML_lnv2a,Liquidity,4.9,0.467,+/-,3.9,4.1,6.3,5.6,1.7
ML_lnv2a,Size,2.8,-0.064,-,4.2,3.0,2.4,1.5,-2.7
ML_lnv2a,CAPMbeta,1.5,0.184,-,2.9,1.2,1.2,1.0,-1.9
ML_lnv2a,Capitalization,0.8,0.190,+/-,0.9,0.7,0.7,0.8,-0.1
ML_lnv2a,Payout,0.4,-0.017,-,0.4,0.4,0.3,0.5,0.1
ML_lnv2s,Efficiency,36.5,-0.677,-,39.0,33.5,39.1,35.0,-4.0
ML_lnv2s,Profitability,23.4,0.346,+,22.9,24.5,20.8,25.7,2.9
ML_lnv2s,Growth,12.4,0.092,+,9.4,14.2,12.9,12.4,3.0
ML_lnv2s,Financial soundness,8.1,0.177,+/-,7.8,8.2,7.7,9.1,1.3
ML_lnv2s,Industry,5.9,0.206,NA,5.0,5.8,6.7,6.0,1.0
ML_lnv2s,Solvency,4.9,-0.161,+/-,5.8,5.1,4.7,3.9,-1.9
ML_lnv2s,Liquidity,3.3,-0.146,+/-,3.0,3.2,3.7,3.4,0.4
ML_lnv2s,Size,3.1,0.090,-,4.2,3.4,2.5,2.2,-2.0
ML_lnv2s,CAPMbeta,1.1,-0.118,-,1.9,1.0,0.9,0.8,-1.1
ML_lnv2s,Capitalization,0.8,-0.171,+/-,0.8,0.7,0.7,1.0,0.1
ML_lnv2s,Payout,0.3,-0.026,-,0.3,0.3,0.2,0.5,0.2
```

Compare our decades with the paper's columns `d1990`, `d2000`, `d2010`.

### 6.6 Table 9 — importance of individual variables over time (%)

```csv
model,variable,category,shap,d1980,d1990,d2000,d2010,chg
ML_lnm2b,profitability_an,Profitability,6.0,2.8,6.7,5.8,9.0,6.2
ML_lnm2b,rd_sale_wr,Growth,4.9,0.0,4.5,7.3,8.1,8.1
ML_lnm2b,industry,Industry,8.9,11.1,8.7,8.3,7.4,-3.7
ML_lnm2b,roe_wr,Profitability,7.7,9.4,7.2,8.7,5.1,-4.3
ML_lnm2b,roic_an,Profitability,2.9,1.1,4.0,2.6,3.4,2.4
ML_lnm2b,gprof_wr,Profitability,1.4,0.9,0.6,1.3,3.2,2.4
ML_lnm2b,roce_wr,Profitability,2.4,0.7,1.7,4.5,2.8,2.1
ML_lnm2b,aftret_invcapx_wr,Profitability,3.7,6.1,3.5,2.6,2.4,-3.7
ML_lnm2b,CAPMbeta,CAPMbeta,2.1,3.4,1.8,1.8,1.3,-2.2
ML_lnm2b,chbe_an,Growth,2.1,3.0,2.9,1.2,0.6,-2.4
ML_lnv2a,sales2rec_an,Efficiency,3.9,1.0,3.0,4.9,7.5,6.5
ML_lnv2a,roic_an,Profitability,7.0,2.9,8.3,8.8,7.2,4.3
ML_lnv2a,industry,Industry,7.7,8.8,7.1,8.5,6.3,-2.5
ML_lnv2a,rd_sale_wr,Growth,3.8,0.0,3.7,5.7,6.2,6.1
ML_lnv2a,rect_turn_wr,Efficiency,1.8,0.5,0.4,2.9,4.4,3.9
ML_lnv2a,roa_wr,Profitability,1.8,0.4,1.5,2.5,3.3,2.9
ML_lnv2a,chbe_an,Growth,4.3,5.8,6.4,2.9,1.1,-4.7
ML_lnv2a,CAPMbeta,CAPMbeta,1.5,2.9,1.2,1.2,1.0,-1.9
ML_lnv2a,lt_ppent_wr,Financial soundness,4.1,5.1,7.5,1.7,0.5,-4.6
ML_lnv2a,Accrual_wr,Financial soundness,2.3,8.5,0.3,0.4,0.5,-8.0
ML_lnv2s,at_turn_wr,Efficiency,9.0,7.5,7.0,9.4,13.9,6.4
ML_lnv2s,opleverage_an,Efficiency,21.2,26.7,20.9,22.7,12.9,-13.9
ML_lnv2s,gpm_wr,Profitability,2.1,1.8,1.1,2.1,4.3,2.6
ML_lnv2s,ebitda2revenue_an,Profitability,1.5,0.5,0.9,1.7,3.6,3.1
ML_lnv2s,rd_sale_wr,Growth,2.5,0.0,2.7,3.8,3.3,3.3
ML_lnv2s,sales2cash_an,Efficiency,2.0,0.8,1.4,2.8,3.3,2.4
ML_lnv2s,npm_wr,Profitability,3.4,5.8,2.9,2.7,2.1,-3.8
ML_lnv2s,chbe_an,Growth,2.1,2.2,3.1,1.7,1.0,-1.3
ML_lnv2s,CAPMbeta,CAPMbeta,1.1,1.9,1.0,0.9,0.8,-1.1
ML_lnv2s,sale_ac,Size,1.1,2.7,0.9,0.4,0.4,-2.3
```

### 6.7 Table 5 — Moderna's peers, December 2019 (paper)

```csv
group,company,lnm2b,weight,cumweight,ff12,ff49,rd_sale_wr,cash2assets_an,de_ratio_wr,roe_wr
test,Moderna Inc,1.66,,,Hlth,DRUGS,6.47,0.67,0.27,-0.42
most,bluebird bio Inc,1.19,0.07,0.07,Hlth,DRUGS,10.02,0.62,0.29,-0.49
most,Guardant Health Inc,2.20,0.05,0.12,Hlth,DRUGS,0.42,0.55,0.11,-0.09
most,Amicus Therapeutics Inc,1.50,0.05,0.17,Hlth,DRUGS,0.77,0.58,0.61,-0.62
most,Liveramp Holdings Inc,1.01,0.04,0.21,BusEq,SOFTW,0.31,0.58,0.15,-0.12
most,Sangamo Therapeutics Inc,0.82,0.04,0.25,Hlth,DRUGS,1.89,0.56,0.53,-0.28
most,MacroGenics Inc,0.74,0.04,0.29,Hlth,DRUGS,3.47,0.72,0.38,-0.65
most,Agios Pharmaceuticals Inc,1.99,0.04,0.33,Hlth,DRUGS,3.54,0.65,0.56,-0.89
most,BridgeBio Pharma Inc,2.24,0.04,0.36,Hlth,DRUGS,7.62,0.84,0.25,-0.49
most,Insmed Inc,1.95,0.03,0.40,Hlth,DRUGS,1.38,0.70,1.51,-0.96
most,TiVo Corp,-0.21,0.03,0.43,BusEq,SOFTW,0.24,0.11,0.96,-0.36
least,Cardlytics Inc,2.57,-0.01,0.98,BusEq,SOFTW,0.07,0.49,0.52,-0.25
least,Phreesia Inc,2.23,-0.01,1.00,BusEq,SOFTW,0.15,0.61,0.58,-0.23
least,Inspire Medical Systems Inc,2.50,-0.01,1.01,Hlth,MEDEQ,0.16,0.83,0.25,-0.20
least,Cryoport Inc,1.72,-0.01,1.02,Hlth,MEDEQ,0.11,0.69,0.22,-0.18
least,Roku Inc,3.30,-0.01,1.03,Durbl,TOYS,0.24,0.43,0.93,-0.08
least,Tandem Diabetes Care Inc,3.05,-0.01,1.04,Hlth,MEDEQ,0.13,0.54,0.75,-0.14
least,Q2 Holdings Inc,2.32,-0.01,1.05,BusEq,SOFTW,0.24,0.65,1.56,-0.18
least,Acorda Therapeutics Inc,-1.07,-0.01,1.06,Hlth,DRUGS,0.37,0.29,2.13,-1.16
least,Catalyst Pharmaceutical Inc,1.60,-0.01,1.07,Hlth,DRUGS,0.29,0.79,0.24,0.12
least,Bandwidth Inc,1.48,-0.01,1.08,BusEq,SOFTW,0.14,0.60,0.19,0.01
```

Which block Moderna falls in depends on the random split, so the peer list will differ. Compare the pattern (mostly drug developers with heavy R&D, large cash balances, low debt, very negative ROE), not the names.

---

## 7. How to judge the results

**Qualitative findings that should hold** (they are about the method, not the sample):

1. The GBMs have the lowest `med(|ε|)` of all models for each multiple, and `med(ε)` close to zero.
2. The GBM is below IND and BL in `med(|ε|)` in almost every month.
3. In Table 4 the ordering of the steps holds: logs help, the GBM helps more, extra variables matter mainly for v2s, and the GBM beats LASSO on complete cases.
4. SHAP: Profitability and Growth are the top categories for m2b and v2a; Efficiency is first for v2s, led by `opleverage_an` and `at_turn_wr` with negative `rho`; ROE-type variables lead m2b, ROA/ROIC-type lead v2a.
5. Peer weights reproduce every prediction exactly and sum to one.
6. Long–short portfolios on GBM errors have positive mean returns. Significance may be weaker than in the paper (shorter period, survivor sample): report what you find.

**Numerical comparison.** For every paper number in section 6 report: paper value, our value, difference, and the seed spread where relevant. Working thresholds proposed by the team (flag a result outside them, investigate, then document; do not force it):

- `med(|ε|)` of each GBM within ±0.05 of the paper's (0.29, 0.30, 0.32).
- `|med(ε)|` of each GBM below 0.03.
- `R²oos(M̂)` of each GBM at least 0.70.
- GBM below IND and BL in at least 90% of months.

**If a number is off, check in this order before calling it a data difference:**

| Symptom | Usual cause |
|---|---|
| `med(|ε|)` far above 0.32 | Units (USD vs millions), wrong `D` or `B`, a miscomputed variable. Recheck the coverage and median table against Table IA.4 |
| `med(|ε|)` far below 0.20 | Leakage: a feature contains the price, the test block is used for early stopping or filtering, duplicated firm-months across blocks |
| `med(ε)` far from zero | Error converting `exp(ŷ)` to equity value (debt subtracted wrongly) |
| GBM does not beat IND or BL | Benchmark fitted wrongly, or the GBM stopped too early (check `best_iteration`) |
| Meaningless SHAP ranking | Columns misaligned between training and `pred_contrib`, or variables miscomputed |

Expected directions of our gaps, to help you tell data effects from bugs: fewer firms per model (about 670 training firms instead of about 2,265) should raise errors somewhat, more so in the 1990s; a survivor sample is tilted to larger, more successful firms, which are easier to value, and should lower them; dollar RMSE depends on the size of the largest firms and on the period, so it is the least comparable number.

---

## 8. Sensitivity runs required by the data situation

Report these next to the headline results so a reader can see how much each data limitation matters:

1. Results by decade and by year, since the sample is thinnest and most biased early on.
2. Listed-today firms only against all firms (`origen`), to show how little the 139 delisted firms change.
3. `industry` categorical against numeric (section 4.1).
4. `CAPMbeta` minimum of 24, 36, 48 and 60 months.
5. The definition switches in the panel builder's header (`RESTAR_DP_DE`, `XSGA_SUMAR_XRD`, `EBIT_CONSTRUIDO`, `LT_INCLUYE_MIB`, `DEUDA_VACIA_ES_CERO`, `NOPI_APROXIMADO`): document the value used and why, using `lseg_prueba5_equivalencias.csv`. In that check `ebit / (sale − cogs − xsga)` is 1.00 and `ebit / (sale − cogs − xsga − dp)` is 1.79, which means LSEG's `cogs`/`xsga` already contain depreciation. Compustat's `cogs` and `xsga` exclude it. Evaluate whether subtracting `dp` from one of them brings gross-margin-type variables closer to the paper's coverage and medians, choose with the Table IA.4 comparison only (never with model errors), and record the decision.
6. Seeds (section 4.1).

---

## 9. Repository organisation

Turn the folder into a monorepo. Do not delete originals: move them. Proposed layout (adapt names only with a recorded reason):

```
.
├── README.md                     how to install, run the replication, read the results
├── Makefile                      setup, panel, train, benchmarks, evaluate, shap, peers, report, test, lint
├── pyproject.toml                workspace config, ruff, pytest
├── .gitignore  .env.example  docker-compose.yml
├── docs/
│   ├── paper/                    the PDF, the Word plan, CAMBIOS_v3.md, CAMBIOS_v4.md
│   ├── replication/              PROGRESS.md DECISIONS.md DEVIATIONS.md BLOCKERS.md REPLICATION_REPORT.md
│   │   └── paper_targets/        section 6 as CSV
│   └── adr/                      short architecture decision records
├── data/                         git-ignored
│   ├── raw/lseg/                 DESCARGAS zips, lseg_datos_mercado.zip, datos_lseg CSVs
│   ├── raw/wrds/                 empty, for a future exact replication
│   ├── external/                 Ken French files, TNIC if obtained
│   ├── interim/  processed/      panel parquet
│   ├── predictions/              master predictions table, partitioned by model
│   └── artifacts/                trained models, SHAP values, peer weights
├── ml/                           the Python package `relval`
│   ├── pyproject.toml
│   ├── configs/                  replication.yaml (period, seeds, paths, switches), one file per experiment
│   ├── src/relval/
│   │   ├── data/                 source adapters (lseg.py, wrds.py stub), schemas, validation
│   │   ├── panel/                accounting records, lags, book equity, filters (refactor of the legacy script)
│   │   ├── features/             the 97 variables, categories registry, beta, industry
│   │   ├── targets/              multiples, conversion to equity value, valuation error
│   │   ├── splits/               five-block monthly split
│   │   ├── models/               gbm.py, benchmarks/{ind,bl,hp,rrv,bg}.py, ols_log.py, lasso.py, cart.py
│   │   ├── evaluation/           metrics, by_time, by_group, portfolios, reconciliation, common_sample
│   │   ├── explain/              shap_values.py, peer_weights.py
│   │   ├── reporting/            tables and figures, paper-vs-ours comparison
│   │   └── pipelines/            CLI entry points, one per stage, idempotent and resumable
│   └── tests/                    unit, method-fidelity and leakage tests
├── backend/                      FastAPI service
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/routers/       health, replication, firms, valuations, explanations, peers, scenarios
│   │   ├── core/                 settings, logging, errors
│   │   ├── schemas/              request and response models
│   │   ├── services/             business logic (valuation, explanation, peers, scenario)
│   │   ├── repositories/         read access to parquet/artifacts (local disk or S3)
│   │   └── ml/                   thin adapter that imports `relval`
│   ├── tests/  Dockerfile  pyproject.toml
├── frontend/                     React + TypeScript + Vite
│   ├── src/{pages,components,features,api,hooks,types}/
│   ├── package.json  Dockerfile
├── terraform/
│   ├── modules/                  network, storage, ecr, iam, training_batch, api_service, frontend_hosting, observability, secrets
│   └── environments/{dev,prod}/  main.tf variables.tf outputs.tf providers.tf backend.tf terraform.tfvars.example
├── scripts/                      one-off helpers
├── notebooks/                    exploration only; nothing the pipeline depends on
├── legacy/                       construir_97_variables.py, unir_lotes_lseg.py, the download notebooks, the Excel dictionaries
└── .github/workflows/            lint + tests (no data needed)
```

**ML package rules.** Typed, small functions; one config file drives a run; every stage reads and writes parquet with a fixed schema; every stage can be rerun and skips finished work; every run writes a manifest (config, git commit, library versions, seeds, row counts, timings). Parallelise by month. On this sample the whole replication is small: thousands of models with about 700 training rows each. Run it locally first.

**Backend (after the replication passes).** A read-only API over the replication outputs, so the later product has a base: `GET /health`; `GET /api/v1/replication/tables/{table_id}` (paper vs ours); `GET /api/v1/firms?query=`; `GET /api/v1/valuations/{id}?date=&target=` (actual and predicted multiple, predicted equity value, valuation error); `GET /api/v1/valuations/{id}/shap`; `GET /api/v1/valuations/{id}/peers`; `POST /api/v1/scenarios` returning 501 with the planned contract (the what-if use case comes later). Routers contain no logic; services contain no I/O details; repositories hide where the data live.

**Frontend (after the backend).** Two pages are enough for now: a replication dashboard (each paper table next to ours, the monthly error figure) and a firm explorer (pick a firm and month: the three valuations, the SHAP waterfall, the top peers). Typed API client, no business logic in components.

**Terraform (AWS).** Modules, each with `variables.tf`, `main.tf`, `outputs.tf` and a short README:

| Module | Contents |
|---|---|
| `network` | VPC, private and public subnets, endpoints for S3 and ECR |
| `storage` | S3 buckets for raw data, processed data and artifacts: versioned, encrypted, all public access blocked, lifecycle rules |
| `ecr` | Image repositories for the training job and the API |
| `iam` | Least-privilege roles for the training job, the API task and CI |
| `training_batch` | AWS Batch compute environment (Spot), job queue and job definition; the pipeline runs as an array job, one child per chunk of months |
| `api_service` | ECS Fargate service behind a load balancer for the backend |
| `frontend_hosting` | S3 + CloudFront for the built frontend |
| `observability` | CloudWatch log groups, alarms, a monthly budget alarm |
| `secrets` | SSM Parameter Store / Secrets Manager entries (names only, no values) |

`environments/dev` and `environments/prod` compose the modules with different sizes and names. Remote state in S3 with locking, configured but not initialised. Region is a variable. No secrets or account IDs in the code. The licensed data may only ever go to the private raw-data bucket of the owner's account, and only when the owner does it.

---

## 10. Method-fidelity tests (must all pass)

Write these as automated tests. They do not depend on the paper's sample, so they must pass exactly.

1. **Panel regression test.** The refactored panel builder gives the same row counts as section 2.3 and the same values as the legacy script on a fixed sample of firm-months.
2. **Formula tests.** Each of the 97 variables against a hand-computed value on a small synthetic firm, including the missing-value rules, the three-month lag, the four-quarter sums and the `l_` lags.
3. **Split tests.** Each month: five disjoint blocks covering all observations, sizes differing by at most one; each observation is in the test block exactly once; train, validation and test never overlap; the split is identical for every model and benchmark and reproducible from the seed.
4. **No-leakage tests.** No feature is computed from `M`, `mv` or price (CAPMbeta uses returns up to t−1 only); accounting data used in month `t` have `datadate` at least three months earlier; industry statistics in IND and BL use the fitting sample only; a permutation check (shuffle the target within month: out-of-sample R² must fall to about zero).
5. **Conversion tests.** `ε` for m2b equals `exp(ŷ − y) − 1`; for v2a and v2s the debt is subtracted as in section 3.2; worked example: `B = 2,000`, `M = 6,000`, `ŷ = 0.95` gives `M̂ ≈ 5,171` and `ε ≈ −13.8%`.
6. **GBM toy example (paper, Table A.2).** Training firms A–E with `lnm2b` 3.6, 3.4, 2.6, 2.2, 1.8; tree 1 puts A, C, D, E in one leaf and B in the other; tree 2 puts A, C in one leaf and B, D, E in the other; learning rate 0.1. The predictions after iterations 0, 1, 2 are 2.72, 2.703, 2.743 for firm A (and for the test firm X, which follows A's leaves).
7. **Peer-weight tests.** On that toy example the weights of X on A, B, C, D, E are 0.235, 0.162, 0.235, 0.185, 0.185 and reproduce 2.743. On real December 2019 models: the weights of every test firm sum to 1 and `vᵀK` equals the LightGBM prediction to numerical precision.
8. **SHAP tests.** Raw SHAP values plus the base value equal the prediction; scaled absolute values sum to 100% per observation; category sums equal the sum of their variables.
9. **Metric tests.** Each metric against a hand calculation; `R²oos` uses the training mean of the model that made the prediction.
10. **Portfolio tests.** Breakpoints come from NYSE firms only; weights use month-`t` equity value; returns are month `t+1`; Newey–West with 3 lags matches `statsmodels`.

---

## 11. Work plan and Definition of Done

Work through the phases in order. A phase is finished when its checks pass and `PROGRESS.md` says so.

| Phase | Output |
|---|---|
| 0. Set-up | Monorepo skeleton (section 9), environment, git, originals moved, `paper_targets/` transcribed and verified, external factor files downloaded |
| 1. Panel | `relval` panel stage reproducing section 2.3; coverage/median table against Table IA.4; Table 1 comparison; tests 1–2 |
| 2. GBM | 5,400 base models, master predictions table, Table 2 rows for the three GBMs, seed spread; tests 3–5 |
| 3. Benchmarks | IND, BL, RRV, BG (approximate), HP (or blocker); full Table 2; common-sample table (IA.5); Figures 1–3 |
| 4. Reconciliation and robustness | Table 4; CART; alternative targets; 80:20 split; section 8 sensitivity runs |
| 5. Portfolios | Table 3 (value-weighted) and IA.7 (equal-weighted) with FF6; 13-factor model or blocker; test 10 |
| 6. Explainability | Tables 7, 8, 9 and the full 97-variable importance table (IA.8); Figure 5; peer weights, Table 5, Figure 4; tests 6–8 |
| 7. Report | `REPLICATION_REPORT.md`, `DEVIATIONS.md`, `DECISIONS.md`, `BLOCKERS.md`, all tables as CSV and Markdown, all figures as PNG, one command (`make replicate`) that regenerates everything from `data/raw/` |
| 8. Backend, frontend, Terraform | Section 9: API over the outputs with tests, the two frontend pages, Terraform that passes `fmt` and `validate` |

**Definition of Done**

- [ ] All tests of section 10 pass, and CI runs lint and tests without needing the data.
- [ ] `make replicate` rebuilds the panel, every model, every table and every figure from `data/raw/`, deterministically for a fixed seed.
- [ ] Every table and figure of the paper's main text (Tables 1–5 and 7–9, Figures 1–5) has our version, generated by code, next to the paper's values from section 6. Online-appendix items covered: summary statistics (IA.4), common sample (IA.5), 80:20 split (IA.6), equal-weighted portfolios (IA.7), full SHAP table (IA.8), CART and alternative targets.
- [ ] `REPLICATION_REPORT.md` states, for each of the six qualitative findings of section 7, whether it holds on our data, with the numbers, and gives the threshold results.
- [ ] `DEVIATIONS.md` lists every difference from the paper in data and method (including each "paper silent" default used), with its measured or expected effect.
- [ ] `BLOCKERS.md` lists what could not be done (HP, 13-factor model, anything else) and exactly what is needed to do it.
- [ ] The `wrds` source adapter is specified and stubbed, so the same pipeline can run on CRSP/Compustat 1980–2019 without other changes.
- [ ] The repository follows section 9; legacy files are preserved under `legacy/`; no data file is tracked by git.
- [ ] The backend serves the replication outputs, the frontend shows them, and the Terraform validates. Nothing was applied to AWS.

When all boxes are ticked, write a final summary at the top of `PROGRESS.md`: what was reproduced, the headline numbers against the paper's, the open blockers, and the exact commands to rerun everything.
