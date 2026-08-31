"""
Assemble the market-level cross-section and the market-year panel.

A "market" is an active-ingredient set crossed with a coarse dosage-form class,
e.g. CARBOPLATIN x INJECTABLE. That is the level at which a shortage actually
bites: a hospital that cannot get one firm's carboplatin vial substitutes
another firm's carboplatin vial, not carboplatin tablets.

Outputs:
  data/derived/markets.csv       - cross-section, one row per market
  data/derived/market_years.csv  - market x year panel, 2012-2026
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd

DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

SNAPSHOT = pd.Timestamp("2026-08-22")
PANEL_START, PANEL_END = 2012, 2026


def load():
    p = pd.read_csv(os.path.join(DERIVED, "products.csv"),
                    parse_dates=["approval_date", "last_ap_date"])
    n = pd.read_csv(os.path.join(DERIVED, "ndc_active.csv"),
                    parse_dates=["marketing_start", "marketing_end"])
    s = pd.read_csv(os.path.join(DERIVED, "shortage_records.csv"),
                    parse_dates=["initial_posting", "update_date",
                                 "discontinued_date", "change_date"])
    for df, a, b in ((p, "ingredient_key", "form_class"),
                     (n, "ingredient_key", "form_class")):
        df["mkt"] = df[a] + " | " + df[b]
    s["mkt"] = s["market_ingredient"] + " | " + s["market_form"]
    return p, n, s


def epc_by_market():
    """Modal Established Pharmacologic Class per market, from the NDC file.

    Coverage is only ~38% of Rx listings, so this enters the models as a
    coarse control with an explicit 'unknown' level rather than as a
    fully-populated fixed effect.
    """
    prods = json.load(open(os.path.join(RAW, "ndc.json")))["results"]
    import build_universe as bu
    rows = []
    for x in prods:
        if x.get("product_type") != "HUMAN PRESCRIPTION DRUG":
            continue
        epc = (x.get("openfda") or {}).get("pharm_class_epc") or []
        if not epc:
            continue
        ings = [i.get("name") for i in x.get("active_ingredients", []) or []]
        ik = bu.ingredient_key(ings)
        if not ik:
            continue
        fc = bu.form_class(x.get("dosage_form"), ", ".join(x.get("route") or []))
        rows.append({"mkt": ik + " | " + fc, "epc": epc[0]})
    df = pd.DataFrame(rows)
    return df.groupby("mkt")["epc"].agg(lambda c: c.value_counts().index[0])


def build_cross_section(p, n, s):
    g = p.groupby("mkt")
    live = p[p.marketing_status.isin(["Prescription", "Over-the-counter"])]
    anda = p[p.app_type == "ANDA"]
    nda = p[p.app_type.isin(["NDA", "BLA"])]

    m = pd.DataFrame(index=sorted(p.mkt.unique()))
    m["form_class"] = [i.split(" | ")[1] for i in m.index]
    m["injectable"] = (m.form_class == "INJECTABLE").astype(int)

    m["n_products_ever"] = g.size()
    m["n_sponsors_ever"] = g["sponsor_norm"].nunique()
    m["n_sponsors_live"] = live.groupby("mkt")["sponsor_norm"].nunique()
    m["n_anda_sponsors"] = anda.groupby("mkt")["sponsor_norm"].nunique()
    m["n_nda_sponsors"] = nda.groupby("mkt")["sponsor_norm"].nunique()
    m["n_disc_products"] = p[p.marketing_status == "Discontinued"].groupby("mkt").size()
    m["first_approval"] = g["approval_date"].min()
    m["first_anda"] = anda.groupby("mkt")["approval_date"].min()

    # current market presence, from the NDC directory
    m["n_labelers_active"] = n.groupby("mkt")["labeler_norm"].nunique()
    m["n_ndc_active"] = n.groupby("mkt").size()
    ndc_anda = n[n.marketing_category.isin(["ANDA", "NDA AUTHORIZED GENERIC"])]
    m["n_labelers_generic"] = ndc_anda.groupby("mkt")["labeler_norm"].nunique()

    for c in ["n_sponsors_live", "n_anda_sponsors", "n_nda_sponsors",
              "n_disc_products", "n_labelers_active", "n_ndc_active",
              "n_labelers_generic"]:
        m[c] = m[c].fillna(0).astype(int)

    m["age_years"] = (SNAPSHOT - m["first_approval"]).dt.days / 365.25
    m["generic_years"] = (SNAPSHOT - m["first_anda"]).dt.days / 365.25
    m["has_generic"] = (m.n_anda_sponsors > 0).astype(int)
    m["generic_only"] = ((m.n_labelers_generic > 0) &
                         (m.n_labelers_generic == m.n_labelers_active)).astype(int)
    # share of all products ever approved in this market that are now dead.
    # undated, so it is a stock measure of churn, not an event.
    m["disc_share"] = m.n_disc_products / m.n_products_ever

    # ---- outcomes -------------------------------------------------------
    cur = s[s.status == "Current"]
    tbd = s[s.status == "To Be Discontinued"]
    m["in_shortage"] = m.index.isin(cur.mkt).astype(int)
    m["shortage_onset"] = cur.groupby("mkt")["initial_posting"].min()
    m["n_shortage_recs"] = cur.groupby("mkt").size().reindex(m.index).fillna(0).astype(int)
    m["n_shortage_firms"] = cur.groupby("mkt")["company_norm"].nunique().reindex(m.index).fillna(0).astype(int)
    m["tbd_notice"] = m.index.isin(tbd.mkt).astype(int)
    m["tbd_date"] = tbd.groupby("mkt")["discontinued_date"].min()
    m["elapsed_years"] = (SNAPSHOT - m["shortage_onset"]).dt.days / 365.25
    reason = (cur.dropna(subset=["shortage_reason"])
                 .groupby("mkt")["shortage_reason"]
                 .agg(lambda c: c.value_counts().index[0]))
    m["shortage_reason"] = reason
    ther = (cur.dropna(subset=["therapeutic_category"])
               .groupby("mkt")["therapeutic_category"]
               .agg(lambda c: c.value_counts().index[0]))
    m["therapeutic_category"] = ther

    m["epc"] = epc_by_market()
    m["epc"] = m["epc"].fillna("UNKNOWN")

    # marketed universe: at least one live approved product AND at least one
    # active NDC listing. Markets that exist only on paper cannot go short.
    m["marketed"] = ((m.n_sponsors_live > 0) & (m.n_labelers_active > 0)).astype(int)
    m.index.name = "mkt"
    return m


def build_market_years(p, m):
    """Market x year panel with a time-varying count of approved suppliers.

    Approval dates survive in drugsfda even for products long since
    discontinued, so the cumulative approval count is the one supplier-side
    measure here that is genuinely free of survivorship bias.
    """
    p = p.dropna(subset=["approval_date"]).copy()
    p["yr"] = p.approval_date.dt.year
    first = p.groupby(["mkt", "sponsor_norm"])["yr"].min().reset_index()
    firsta = (p[p.app_type == "ANDA"].groupby(["mkt", "sponsor_norm"])["yr"]
              .min().reset_index())

    years = list(range(PANEL_START, PANEL_END + 1))
    mkts = sorted(m.index[m.marketed == 1])
    idx = pd.MultiIndex.from_product([mkts, years], names=["mkt", "year"])
    panel = pd.DataFrame(index=idx).reset_index()

    def cum(fr, col):
        t = (fr.groupby(["mkt", "yr"]).size().rename(col).reset_index()
               .pivot(index="mkt", columns="yr", values=col)
               .reindex(mkts).fillna(0))
        for y in range(t.columns.min(), PANEL_END + 1):
            if y not in t.columns:
                t[y] = 0
        t = t.sort_index(axis=1).cumsum(axis=1)
        t = t.reindex(columns=years, method="ffill").fillna(0)
        return t.stack().rename(col).reset_index().rename(columns={"yr": "year",
                                                                   "level_1": "year"})

    panel = panel.merge(cum(first, "cum_sponsors"), on=["mkt", "year"], how="left")
    panel = panel.merge(cum(firsta, "cum_anda_sponsors"), on=["mkt", "year"], how="left")
    panel[["cum_sponsors", "cum_anda_sponsors"]] = panel[
        ["cum_sponsors", "cum_anda_sponsors"]].fillna(0)

    static = m.loc[mkts, ["injectable", "form_class", "first_approval",
                          "shortage_onset", "in_shortage", "epc",
                          "n_labelers_active", "generic_only"]]
    panel = panel.merge(static, left_on="mkt", right_index=True, how="left")
    panel["age_years"] = panel.year - panel.first_approval.dt.year
    panel["onset_year"] = panel.shortage_onset.dt.year

    # at risk from the later of panel start and first approval, until the year
    # of first (still-active) shortage onset
    panel = panel[panel.age_years >= 0].copy()
    panel["event"] = (panel.year == panel.onset_year).astype(int)
    panel = panel[panel.onset_year.isna() | (panel.year <= panel.onset_year)].copy()
    return panel


if __name__ == "__main__":
    p, n, s = load()
    m = build_cross_section(p, n, s)
    m.to_csv(os.path.join(DERIVED, "markets.csv"))
    print("markets:", len(m), "| marketed:", int(m.marketed.sum()),
          "| in shortage:", int(m.in_shortage.sum()),
          "| in shortage & marketed:", int((m.in_shortage & m.marketed).sum()))

    mm = m[m.marketed == 1]
    print("\nshortage rate by form class:")
    t = mm.groupby("form_class").agg(n=("in_shortage", "size"),
                                     short=("in_shortage", "sum"))
    t["rate_pct"] = (100 * t.short / t.n).round(2)
    print(t.sort_values("rate_pct", ascending=False).to_string())

    print("\nshortage rate by active-labeler count (injectable vs not):")
    mm = mm.copy()
    mm["nbin"] = pd.cut(mm.n_labelers_active, [0, 1, 2, 4, 8, 1000],
                        labels=["1", "2", "3-4", "5-8", "9+"])
    t = mm.pivot_table(index="nbin", columns="injectable", values="in_shortage",
                       aggfunc=["size", "sum"], observed=True)
    print(t.to_string())

    panel = build_market_years(p, m)
    panel.to_csv(os.path.join(DERIVED, "market_years.csv"), index=False)
    print("\npanel rows:", len(panel), "| markets:", panel.mkt.nunique(),
          "| events:", int(panel.event.sum()))
