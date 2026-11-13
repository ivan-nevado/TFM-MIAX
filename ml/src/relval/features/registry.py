"""Registry of the 97 input variables: categories, descriptions, orientation and duplicate pairs.

Categories follow the paper (Table IA.4 / section 3.3 of the master prompt). ``ORIENTATION`` is
used only for the category-level ``rho`` of Table 8: the paper re-orients ranks "so that a higher
rank conforms to the semantics of the category label" but does not list the variables it flips;
the list below is our reading (documented in DECISIONS.md).
"""

from __future__ import annotations

CATEGORIES: dict[str, list[str]] = {
    "Profitability": [
        "aftret_eq_wr", "aftret_equity_wr", "aftret_invcapx_wr", "cfm_wr", "ebitda2revenue_an",
        "efftax_wr", "gpm_wr", "gprof_wr", "grossmargin_an", "grossprofit_an", "marginch_an",
        "npm_wr", "opmad_wr", "opmbd_wr", "pretret_earnat_wr", "pretret_noa_wr",
        "profitability_an", "ptpm_wr", "roa_wr", "roce_wr", "roe_wr", "roic_an",
    ],
    "Growth": [
        "assetgrowth_an", "bookequitygrowth_an", "capexgrowth_an", "chbe_an", "chca_an",
        "chceq_an", "chcl_an", "chfnl_an", "chlt_an", "chnccl_an", "deprn_an",
        "inventorychange_an", "inventorygrowth_an", "investment_an", "pchdeprn_an",
        "rd_sale_wr", "salesgrowth_an",
    ],
    "Financial soundness": [
        "Accrual_wr", "cash_debt_wr", "cash_lt_wr", "curr_debt_wr", "debt_ebitda_wr",
        "dltt_be_wr", "fcf_ocf_wr", "int_debt_wr", "int_totdebt_wr", "invt_act_wr",
        "lt_debt_wr", "lt_ppent_wr", "ocf_lct_wr", "profit_lct_wr", "rect_act_wr", "short_debt_wr",
    ],
    "Efficiency": [
        "assetturnover2_an", "assetturnover_an", "at_turn_wr", "invturn_wr", "opleverage_an",
        "pay_turn_wr", "pchgm2pchsale_an", "pchsale2pchinvt_an", "pchsale2pchrect_an",
        "pchsale2pchxsga_an", "pchsales2inv_an", "rect_turn_wr", "sales2cash_an",
        "sales2inv_an", "sales2rec_an",
    ],
    "Liquidity": [
        "cash2assets_an", "cash_conversion_wr", "cash_ratio_wr", "chcurrentratio_an",
        "curr_ratio_wr", "liquid2assets_an", "pchquickratio_an", "quick_ratio_wr",
    ],
    "Solvency": [
        "cf2debt_an", "de_ratio_wr", "debt2tang_an", "debt_assets_wr", "debt_at_wr",
        "debt_capital_wr", "intcov_ratio_wr", "intcov_wr",
    ],
    "Capitalization": ["capital_ratio_wr", "debt_invcap_wr", "equity_invcap_wr", "totdebt_invcap_wr"],
    "Size": ["assets", "book", "debt_book_value", "sale_ac"],
    "Industry": ["industry"],
    "Payout": ["dpr_wr"],
    "CAPMbeta": ["CAPMbeta"],
}  # fmt: skip

CATEGORY_ORDER = list(CATEGORIES)
FEATURES: list[str] = [v for members in CATEGORIES.values() for v in members]
CATEGORY_OF: dict[str, str] = {v: cat for cat, members in CATEGORIES.items() for v in members}
ACCOUNTING_FEATURES = [f for f in FEATURES if f not in ("industry", "CAPMbeta")]
assert len(FEATURES) == 97 and len(set(FEATURES)) == 97, "the paper uses exactly 97 input variables"

# Same formula computed twice (one WRDS ratio, one anomaly variable). Kept as separate model
# inputs, as in the paper; SHAP tables also report each pair summed.
DUPLICATE_PAIRS: list[tuple[str, str]] = [
    ("gpm_wr", "grossmargin_an"),
    ("gprof_wr", "grossprofit_an"),
    ("opmbd_wr", "ebitda2revenue_an"),
    ("at_turn_wr", "assetturnover_an"),
]

# Variables whose rank is reversed for the category rho (higher value = less of the label).
REVERSED: set[str] = {
    # Financial soundness: more debt, more accruals, more short-term pressure = less sound
    "Accrual_wr", "curr_debt_wr", "debt_ebitda_wr", "dltt_be_wr", "int_debt_wr",
    "int_totdebt_wr", "lt_debt_wr", "lt_ppent_wr", "short_debt_wr",
    # Solvency: more leverage = less solvent
    "de_ratio_wr", "debt_assets_wr", "debt_at_wr", "debt_capital_wr",
    # Capitalization: more debt in invested capital = weaker capitalisation
    "capital_ratio_wr", "debt_invcap_wr", "totdebt_invcap_wr",
}  # fmt: skip
ORIENTATION: dict[str, int] = {f: (-1 if f in REVERSED else 1) for f in FEATURES}

# Expected sign of the marginal effect (paper, Table 6 and section 5.1). "v2s" overrides apply.
EXPECTED_SIGN: dict[str, str] = {
    "Profitability": "+", "Growth": "+", "Efficiency": "+", "Financial soundness": "+/-",
    "Solvency": "+/-", "Liquidity": "+/-", "Capitalization": "+/-", "Size": "-",
    "CAPMbeta": "-", "Payout": "-", "Industry": "NA",
}  # fmt: skip
EXPECTED_SIGN_OVERRIDE: dict[tuple[str, str], str] = {("lnv2s", "Efficiency"): "-"}

DESCRIPTIONS: dict[str, str] = {
    "Accrual_wr": "Accruals/Total Assets", "CAPMbeta": "CAPM beta (60 months)",
    "aftret_eq_wr": "After-tax Return on Common Equity",
    "aftret_equity_wr": "After-tax Return on Stockholders' Equity",
    "aftret_invcapx_wr": "After-tax Return on Invested Capital", "assetgrowth_an": "Asset Growth",
    "assets": "Total book assets", "assetturnover2_an": "Asset Turnover (average assets)",
    "assetturnover_an": "Asset Turnover", "at_turn_wr": "Asset Turnover",
    "book": "Book equity (Davis, Fama & French 2000)", "bookequitygrowth_an": "Growth in Book Equity",
    "capexgrowth_an": "Capex Growth", "capital_ratio_wr": "Capitalisation Ratio",
    "cash2assets_an": "Cash/Total Assets", "cash_conversion_wr": "Cash Conversion Cycle (Days)",
    "cash_debt_wr": "Cash Flow/Total Debt", "cash_lt_wr": "Cash Balance/Total Liabilities",
    "cash_ratio_wr": "Cash Ratio", "cf2debt_an": "Cash Flow/Total Liabilities",
    "cfm_wr": "Cash Flow Margin", "chbe_an": "Change in Book Equity",
    "chca_an": "Change in Noncash Current Assets", "chceq_an": "Change in Common Equity",
    "chcl_an": "Change in Nondebt Current Liabilities", "chcurrentratio_an": "Change in Current Ratio",
    "chfnl_an": "Change in Financial Liabilities", "chlt_an": "Change in Total Liabilities",
    "chnccl_an": "Change in Noncurrent Operating Liabilities",
    "curr_debt_wr": "Current Liabilities/Total Liabilities", "curr_ratio_wr": "Current Ratio",
    "de_ratio_wr": "Total Debt/Equity", "debt2tang_an": "Debt Capacity to Tangibility",
    "debt_assets_wr": "Total Liabilities/Total Assets", "debt_at_wr": "Long-term Debt/Total Assets",
    "debt_book_value": "Book value of debt", "debt_capital_wr": "Total Debt/Capital",
    "debt_ebitda_wr": "Total Debt/EBITDA", "debt_invcap_wr": "Long-term Debt/Invested Capital",
    "deprn_an": "Depreciation Rate", "dltt_be_wr": "Long-term Debt to Equity",
    "dpr_wr": "Dividend Payout Ratio", "ebitda2revenue_an": "EBITDA Margin",
    "efftax_wr": "Effective Tax Rate", "equity_invcap_wr": "Common Equity/Invested Capital",
    "fcf_ocf_wr": "Free Cash Flow/Operating Cash Flow", "gpm_wr": "Gross Profit Margin",
    "gprof_wr": "Gross Profit/Total Assets", "grossmargin_an": "Gross Profit Margin",
    "grossprofit_an": "Gross Profit/Total Assets", "industry": "Fama-French 49 industries",
    "int_debt_wr": "Interest/Long-term Debt", "int_totdebt_wr": "Interest/Total Debt",
    "intcov_ratio_wr": "Interest Coverage Ratio", "intcov_wr": "After-tax Interest Coverage",
    "inventorychange_an": "Change in Inventory", "inventorygrowth_an": "Inventory Growth",
    "investment_an": "Investment", "invt_act_wr": "Inventory/Current Assets",
    "invturn_wr": "Inventory Turnover", "liquid2assets_an": "Liquid Assets",
    "lt_debt_wr": "Long-term Debt/Total Liabilities", "lt_ppent_wr": "Total Liabilities/PP&E",
    "marginch_an": "Change in Profit Margin", "npm_wr": "Net Profit Margin",
    "ocf_lct_wr": "Operating Cash Flow/Current Liabilities", "opleverage_an": "Operating Leverage",
    "opmad_wr": "EBIT Margin", "opmbd_wr": "EBITDA Margin", "pay_turn_wr": "Payables Turnover",
    "pchdeprn_an": "Change in Depreciation Rate", "pchgm2pchsale_an": "Change in Gross Profit - Sales",
    "pchquickratio_an": "Change in Quick Ratio", "pchsale2pchinvt_an": "Change in Sales - Inventory",
    "pchsale2pchrect_an": "Change in Sales - Receivables", "pchsale2pchxsga_an": "Change in Sales - SGA",
    "pchsales2inv_an": "Change in Sales/Inventory",
    "pretret_earnat_wr": "Pre-tax Return on Total Earning Assets",
    "pretret_noa_wr": "Pre-tax Return on Net Operating Assets",
    "profit_lct_wr": "EBITDA/Current Liabilities", "profitability_an": "Profitability (pbt/equity)",
    "ptpm_wr": "Pre-tax Profit Margin", "quick_ratio_wr": "Quick Ratio", "rd_sale_wr": "R&D/Sales",
    "rect_act_wr": "Receivables/Current Assets", "rect_turn_wr": "Receivables Turnover",
    "roa_wr": "Return on Assets", "roce_wr": "Return on Capital Employed",
    "roe_wr": "Return on Equity", "roic_an": "Return on Invested Capital",
    "sale_ac": "Sales/Turnover (Net)", "sales2cash_an": "Sales/Cash",
    "sales2inv_an": "Sales/Inventory", "sales2rec_an": "Sales/Receivables",
    "salesgrowth_an": "Sales Growth", "short_debt_wr": "Short-Term Debt/Total Debt",
    "totdebt_invcap_wr": "Total Debt/Invested Capital",
}  # fmt: skip
assert set(DESCRIPTIONS) == set(FEATURES)


def expected_sign(target: str, category: str) -> str:
    return EXPECTED_SIGN_OVERRIDE.get((target, category), EXPECTED_SIGN[category])
