# Blockers

What could not be done, the evidence, and exactly what would unblock it. None of these stops the
rest of the pipeline.

## 1. HP benchmark (Hoberg & Phillips text-based peers)

- **Status:** implemented (`ml/src/relval/models/benchmarks/hp.py`) but not run. The runner
  records it in `data/artifacts/<run>/status/unavailable.json` and Table 2 has no HP rows.
- **Evidence:** the TNIC files are keyed by Compustat `gvkey` (columns `year, gvkey1, gvkey2,
  score`). The LSEG download identifies firms by RIC, ISIN and CUSIP. TNIC ships no CUSIP or
  ticker, and gvkey↔CUSIP mappings come from Compustat or the CRSP/Compustat link table, which
  need a WRDS subscription.
- **To unblock:** place two files in `data/external/tnic/`:
  - `tnic3_data.txt` from https://hobergphillips.tuck.dartmouth.edu/industryclass.htm
  - `gvkey_link.csv` with columns `gvkey,id` (id = RIC as in the panel), built from a legitimate
    gvkey↔CUSIP source.

  Then run `relval train --models HP_m2b,HP_v2a,HP_v2s` and `relval report`.

## 2. Thirteen-factor model (Table 3, panel "FF6 + 7 clusters of anomalies")

- **Status:** not run. Table 3 reports the mean, Sharpe ratio and FF6 panels.
- **Evidence:** the seven anomaly-cluster factor series of Geertsema & Lu (2020) are not in the
  Ken French library, the paper's online appendix or the authors' public repositories.
- **To unblock:** monthly returns of the seven factors (EG, Accruals, STReversal, MarginGrowth,
  epsconsistency, ExternalFinance, seasonality) from the authors. `evaluation/portfolios.py`
  takes the factor list as a constant (`FF6`); adding the seven columns to the factor table and a
  second regression is a small change.

## 3. Exact replication on the paper's data (CRSP/Compustat, 1980–2019)

- **Status:** the source-adapter interface is in place and the `wrds` adapter is specified and
  stubbed (`ml/src/relval/data/wrds.py`). It raises `NotImplementedError` with instructions.
- **To unblock:** a WRDS subscription. Export `crsp_msf`, `comp_fundq` and `ccm_link` to
  `data/raw/wrds/`, implement the mapping described in the module docstring, set `source: wrds`,
  `period.start: "1980-01"` and the LSEG-specific switches off. Nothing else changes.

## 4. Runs not completed on the development machine

The pipeline is complete and resumable; the full set of model variants is heavy for a laptop
(about 60 variants, several of which need thousands of trees). `PROGRESS.md` lists which groups
have been run locally and which are left for AWS Batch. `relval status` shows the exact state and
`relval replicate` continues from it.

## 5. Table IA.4 of the paper: 17 of 100 rows not parsed

`scripts/paper_targets.py appendix` reads N, mean and median for 83 variables from the online
appendix PDF; the remaining rows have a different text layout. Coverage for all 97 inputs is
available from the team's plan (`ia4_ia8.csv`), so only the paper's mean and median are missing
for those rows. To unblock: extend the parser or transcribe the 17 rows by hand.
