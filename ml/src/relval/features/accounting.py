"""The 95 accounting input variables of Geertsema & Lu (2023), Table A.1.

Input: accounting records from :mod:`relval.panel.records` (balance items of the latest period,
annualised flows, the same with prefix ``l_``) plus the derived quantities of
:func:`relval.panel.book_equity.derive`. Conventions (section 3.3 of the master prompt):

* ``sas_sum`` ignores missing arguments; ``rowtotal`` and ``nz`` treat missing as zero.
* A zero or missing denominator gives missing; an unmet condition ("if be > 0") gives missing.
* Where Table IA.4 reports ~100% coverage for a ratio whose Compustat inputs are often missing
  (``xsga``, ``xint``, ``d_invt``), the missing input is treated as zero, which is what the WRDS
  ratio code does. Each such case is marked ``# zero-fill`` and listed in DECISIONS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.features.registry import ACCOUNTING_FEATURES
from relval.panel.book_equity import derive, div, nz, rowtotal, sas_sum


def compute_accounting_features(
    rec: pd.DataFrame, is_financial: pd.Series, defs: dict | None = None
) -> pd.DataFrame:
    """Return a frame with the 95 accounting variables, aligned with ``rec``.

    ``is_financial`` (bool, aligned with ``rec``) marks SIC 6000-6999 firms, for which
    ``cash2assets_an`` is missing by definition.
    """
    defs = defs or {}
    d = rec.copy()
    d = derive(d, "")
    d = derive(d, "l_")

    def c(name: str) -> pd.Series:
        return d[name]

    def lag(name: str) -> pd.Series:
        return d["l_" + name]

    def chg(name: str) -> pd.Series:  # d_x = x - l_x
        return d[name] - d["l_" + name]

    def where(x: pd.Series, cond: pd.Series) -> pd.Series:
        return x.where(cond)

    at, act, lct, lt, che = c("at"), c("act"), c("lct"), c("lt"), c("che")
    invt, rect, ap, ppent = c("invt"), c("rect"), c("ap"), c("ppent")
    dltt, dlc, pstk, ceq, be, seq = c("dltt"), c("dlc"), c("pstk"), c("ceq"), c("be"), c("seq")
    sale, cogs, xsga, xint, ebit, dp = c("sale"), c("cogs"), c("xsga"), c("xint"), c("ebit"), c("dp")
    ib, ebitda, ocf, capx, icapt = c("ib"), c("ebitda"), c("ocf"), c("capx"), c("icapt")
    l_at, l_sale = lag("at"), lag("sale")
    total_debt = dltt + dlc
    ceq_pstk = sas_sum(ceq, pstk)
    gross = sale - cogs
    l_gross = l_sale - lag("cogs")
    sales_growth = div(sale - l_sale, l_sale)
    # cogs + xsga. In Compustat this equals sale - oibdp (operating cost before depreciation).
    # Vendors that put depreciation inside cogs/xsga and leave cogs blank for financial firms
    # reproduce the Compustat quantity with sale - ebitda (switch ``opcost_from_ebitda``).
    opcost = (sale - ebitda) if defs.get("opcost_from_ebitda", False) else (cogs + nz(xsga))  # zero-fill xsga

    f: dict[str, pd.Series] = {}

    # ---- Profitability -------------------------------------------------------------
    f["aftret_eq_wr"] = div(c("ibcom"), ceq)
    f["aftret_equity_wr"] = div(ib, seq)
    den = sas_sum(lag("icapt"), lag("txditc"), -lag("mib"))
    f["aftret_invcapx_wr"] = where(div(sas_sum(ib + xint, c("mii")), den), den > 0)
    f["cfm_wr"] = div(ib + dp, sale)
    f["ebitda2revenue_an"] = div(ebitda, sale)
    efftax = div(c("txt"), c("pi"))
    f["efftax_wr"] = where(efftax, c("pi") > 0) if defs.get("efftax_positive_pi_only", True) else efftax
    f["gpm_wr"] = div(gross, sale)
    f["gprof_wr"] = div(gross, at)
    f["grossmargin_an"] = div(gross, sale)
    f["grossprofit_an"] = div(gross, at)
    f["marginch_an"] = div(ib, sale) - div(lag("ib"), l_sale)
    f["npm_wr"] = div(ib, sale)
    f["opmad_wr"] = div(ebit, sale)
    f["opmbd_wr"] = div(ebitda, sale)
    f["pretret_earnat_wr"] = div(ebitda, c("earnat"))
    f["pretret_noa_wr"] = div(ebit, c("noa"))
    f["profitability_an"] = div(sale - opcost - nz(xint), be)  # zero-fill xint
    if defs.get("nopi_spi_from_pretax", False):
        f["ptpm_wr"] = div(c("pi"), sale)  # spi + nopi := pi - (ebit - xint)
    else:
        f["ptpm_wr"] = div(ebit - nz(xint) + nz(c("spi")) + nz(c("nopi")), sale)  # zero-fill
    f["roa_wr"] = div(ebitda, at)
    f["roce_wr"] = div(ebit, dltt + pstk + dlc + ceq)
    f["roe_wr"] = where(div(ib, be), be > 0)
    f["roic_an"] = div(ebit - nz(c("nopi")), ceq + lt - che)  # zero-fill nopi

    # ---- Growth --------------------------------------------------------------------
    f["assetgrowth_an"] = div(chg("at"), l_at)
    f["bookequitygrowth_an"] = div(be, lag("be")) - 1.0
    f["capexgrowth_an"] = div(chg("capx"), lag("capx"))
    f["chbe_an"] = div(be - lag("be"), l_at)
    f["chca_an"] = div(chg("act") - chg("che"), l_at)
    f["chceq_an"] = div(ceq - lag("ceq"), lag("ceq"))
    f["chcl_an"] = div(chg("lct") - nz(chg("dlc")), l_at)
    f["chfnl_an"] = div(nz(chg("dltt")) + nz(chg("dlc")) + nz(chg("pstk")), l_at)
    f["chlt_an"] = div(lt, lag("lt")) - 1.0
    f["chnccl_an"] = div(chg("lt") - chg("lct") - nz(chg("dltt")), l_at)
    f["deprn_an"] = div(dp, ppent)
    f["inventorychange_an"] = div(chg("invt"), 0.5 * l_at + 0.5 * at)
    f["inventorygrowth_an"] = div(chg("invt"), lag("invt"))
    f["investment_an"] = div(chg("at"), at)
    l_deprn = div(lag("dp"), lag("ppent"))
    f["pchdeprn_an"] = div(div(dp, ppent) - l_deprn, l_deprn)
    f["rd_sale_wr"] = div(nz(c("xrd")), sale).clip(lower=0.0)
    f["salesgrowth_an"] = div(sale, l_sale) - 1.0

    # ---- Financial soundness ---------------------------------------------------------
    f["Accrual_wr"] = div(ocf - ib, at)
    f["cash_debt_wr"] = div(ocf, lt)
    f["cash_lt_wr"] = div(che, lt)
    f["curr_debt_wr"] = div(lct, lt)
    f["debt_ebitda_wr"] = div(total_debt, ebitda)
    f["dltt_be_wr"] = where(div(dltt, be), be > 0)
    f["fcf_ocf_wr"] = where(div(ocf - capx, ocf), ocf > 0)
    f["int_debt_wr"] = div(xint, dltt)
    f["int_totdebt_wr"] = div(xint, total_debt)
    f["invt_act_wr"] = div(invt, act)
    f["lt_debt_wr"] = div(dltt, lt)
    f["lt_ppent_wr"] = div(lt, ppent)
    f["ocf_lct_wr"] = div(ocf, lct)
    f["profit_lct_wr"] = div(ebitda, lct)
    f["rect_act_wr"] = div(rect, act)
    f["short_debt_wr"] = div(dlc, total_debt)

    # ---- Efficiency ------------------------------------------------------------------
    f["assetturnover2_an"] = div(sale, (at + l_at) / 2.0)
    f["assetturnover_an"] = div(sale, at)
    f["at_turn_wr"] = div(sale, at)
    avg_invt = (invt + lag("invt")) / 2.0
    f["invturn_wr"] = where(div(cogs, avg_invt), avg_invt > 0)
    f["opleverage_an"] = div(opcost, at)
    avg_ap = (ap + lag("ap")) / 2.0
    f["pay_turn_wr"] = where(div(cogs + nz(chg("invt")), avg_ap), avg_ap > 0)  # zero-fill d_invt
    f["pchgm2pchsale_an"] = div(gross - l_gross, l_gross) - sales_growth
    f["pchsale2pchinvt_an"] = sales_growth - div(invt - lag("invt"), lag("invt"))
    f["pchsale2pchrect_an"] = sales_growth - div(rect - lag("rect"), lag("rect"))
    f["pchsale2pchxsga_an"] = sales_growth - div(xsga - lag("xsga"), lag("xsga"))
    l_s2i = div(l_sale, lag("invt"))
    f["pchsales2inv_an"] = div(div(sale, invt) - l_s2i, l_s2i)
    f["rect_turn_wr"] = div(sale, (rect + lag("rect")) / 2.0)
    f["sales2cash_an"] = div(sale, che)
    f["sales2inv_an"] = div(sale, invt)
    f["sales2rec_an"] = div(sale, rect)

    # ---- Liquidity -------------------------------------------------------------------
    f["cash2assets_an"] = where(div(che, at), ~is_financial.to_numpy(dtype=bool))
    f["cash_conversion_wr"] = (
        div(avg_invt, cogs / 365.0)
        + div((rect + lag("rect")) / 2.0, sale / 365.0)
        - div(avg_ap, cogs / 365.0)
    )
    f["cash_ratio_wr"] = where(div(che, lct), lct > 0)
    l_cr = div(lag("act"), lag("lct"))
    f["chcurrentratio_an"] = div(div(act, lct) - l_cr, l_cr)
    f["curr_ratio_wr"] = div(act, lct)
    f["liquid2assets_an"] = div(che + 0.75 * (act - che) + 0.5 * (at - act), l_at)
    l_qr = div(lag("act") - lag("invt"), lag("lct"))
    f["pchquickratio_an"] = div(div(act - invt, lct) - l_qr, l_qr)
    f["quick_ratio_wr"] = where(div(act - invt, lct), lct > 0)

    # ---- Solvency --------------------------------------------------------------------
    f["cf2debt_an"] = div(ib + dp, (lt + lag("lt")) / 2.0)
    f["de_ratio_wr"] = div(lt, ceq_pstk)
    f["debt2tang_an"] = div(che + rect * 0.715 + invt * 0.547 + ppent * 0.535, at)
    f["debt_assets_wr"] = div(lt, at)
    f["debt_at_wr"] = div(total_debt, at)
    debt_plus_ap = ap + sas_sum(dlc, dltt)
    f["debt_capital_wr"] = div(debt_plus_ap, debt_plus_ap + ceq_pstk)
    f["intcov_ratio_wr"] = div(ebit, xint)
    f["intcov_wr"] = div(xint + ib, xint)

    # ---- Capitalization --------------------------------------------------------------
    f["capital_ratio_wr"] = div(dltt, dltt + ceq_pstk)
    f["debt_invcap_wr"] = where(div(dltt, icapt), icapt > 0)
    f["equity_invcap_wr"] = where(div(ceq, icapt), icapt > 0)
    f["totdebt_invcap_wr"] = where(div(total_debt, icapt), icapt > 0)

    # ---- Size and payout ---------------------------------------------------------------
    f["assets"] = at
    f["book"] = be
    f["debt_book_value"] = rowtotal(dltt, dlc, pstk)
    f["sale_ac"] = sale
    dpr = div(c("dv"), c("ibadj"))
    f["dpr_wr"] = where(dpr, c("ibadj") > 0) if defs.get("dpr_positive_income_only", False) else dpr

    out = pd.DataFrame(f, index=rec.index)[ACCOUNTING_FEATURES]
    return out.replace([np.inf, -np.inf], np.nan)
