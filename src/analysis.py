"""
Models.

1. Cross-sectional prevalence of active shortage (fully identified from the
   snapshot: every market's shortage state is observed today).
2. Discrete-time hazard of shortage onset (identified only for shortages that
   persist to the snapshot -- see caveats in the report).
3. Duration of ongoing shortages, with a length-bias correction.

Everything is clustered or stratified where it needs to be, and every
specification prints the coefficient path as controls are added, because the
supplier-count coefficient is the one most sensitive to what you condition on.
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)

SNAPSHOT = pd.Timestamp("2026-08-22")


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def tidy(res, keep=None, label=""):
    df = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    df["or"] = np.exp(df.coef)
    df["out"] = df.apply(
        lambda r: "%.3f (%.3f)%s" % (r.coef, r.se, stars(r.p)), axis=1)
    if keep:
        df = df.loc[[k for k in keep if k in df.index]]
    return df["out"].rename(label)


def load():
    m = pd.read_csv(os.path.join(DERIVED, "markets.csv"), index_col=0,
                    parse_dates=["first_approval", "first_anda",
                                 "shortage_onset", "tbd_date"])
    m = m[m.marketed == 1].copy()
    m["log_labelers"] = np.log(m.n_labelers_active.clip(lower=1))
    m["log_listings"] = np.log(m.n_ndc_active.clip(lower=1))
    m["log_products"] = np.log(m.n_products_ever.clip(lower=1))
    m["log_age"] = np.log(m.age_years.clip(lower=0.5))
    m["single_source"] = (m.n_labelers_active <= 1).astype(int)

    # collapse EPC to the classes with enough mass to estimate, plus catch-alls
    top = m.epc.value_counts()
    top = top[(top >= 25) & (top.index != "UNKNOWN")].index
    m["epc_grp"] = np.where(m.epc.isin(top), m.epc,
                            np.where(m.epc == "UNKNOWN", "UNKNOWN", "OTHER"))
    return m


# ---------------------------------------------------------------------------
# 1. cross-sectional prevalence
# ---------------------------------------------------------------------------

def cross_section(m):
    print("=" * 78)
    print("1. CROSS-SECTION: is this market in an active shortage on 2026-08-22?")
    print("=" * 78)
    print("markets at risk: %d   in shortage: %d (%.2f%%)"
          % (len(m), m.in_shortage.sum(), 100 * m.in_shortage.mean()))

    specs = [
        ("(1) form only", "in_shortage ~ injectable"),
        ("(2) + suppliers", "in_shortage ~ injectable + log_labelers"),
        ("(3) + size/age", "in_shortage ~ injectable + log_labelers + log_listings + log_age"),
        ("(4) + generic", "in_shortage ~ injectable + log_labelers + log_listings + log_age + generic_only"),
        ("(5) + class FE", "in_shortage ~ injectable + log_labelers + log_listings + log_age + generic_only + C(epc_grp)"),
    ]
    keep = ["injectable", "log_labelers", "log_listings", "log_age", "generic_only"]
    cols, models = [], {}
    for lbl, f in specs:
        r = smf.glm(f, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
        cols.append(tidy(r, keep, lbl))
        models[lbl] = r
    tab = pd.concat(cols, axis=1)
    print("\nlogit coefficients (HC1 SE in parens)")
    print(tab.to_string())

    r = models["(5) + class FE"]
    print("\nodds ratios, full spec:")
    for k in keep:
        if k in r.params.index:
            lo, hi = r.conf_int().loc[k]
            print("  %-14s OR %6.2f  [%.2f, %.2f]  p=%.4f"
                  % (k, np.exp(r.params[k]), np.exp(lo), np.exp(hi), r.pvalues[k]))

    # average marginal effect of being injectable
    m0, m1 = m.copy(), m.copy()
    m0["injectable"], m1["injectable"] = 0, 1
    ame = r.predict(m1).mean() - r.predict(m0).mean()
    print("\naverage marginal effect of INJECTABLE: %+.2f pp"
          % (100 * ame))

    # injectables only: does supplier count matter inside the risky segment?
    inj = m[m.injectable == 1]
    r2 = smf.glm("in_shortage ~ log_labelers + log_listings + log_age + generic_only",
                 data=inj, family=sm.families.Binomial()).fit(cov_type="HC1")
    print("\ninjectables only (n=%d, events=%d):" % (len(inj), inj.in_shortage.sum()))
    print(tidy(r2, ["log_labelers", "log_listings", "log_age", "generic_only"],
               "injectable subsample").to_string())
    return tab, models


# ---------------------------------------------------------------------------
# 2. discrete-time hazard of onset
# ---------------------------------------------------------------------------

def hazard(m):
    print("\n" + "=" * 78)
    print("2. DISCRETE-TIME HAZARD of onset of a shortage still active in 2026")
    print("=" * 78)
    pn = pd.read_csv(os.path.join(DERIVED, "market_years.csv"))
    pn = pn[pn.mkt.isin(m.index)].copy()
    pn["log_cum_sponsors"] = np.log(pn.cum_sponsors.clip(lower=1))
    pn["log_age"] = np.log(pn.age_years.clip(lower=0.5))
    # market_years already carries injectable/generic_only/epc from the static
    # merge in build_panel, so only pull what is genuinely new
    pn = pn.merge(m[["log_listings", "epc_grp"]],
                  left_on="mkt", right_index=True, how="left")
    print("market-years: %d   markets: %d   onsets: %d"
          % (len(pn), pn.mkt.nunique(), pn.event.sum()))

    f = ("event ~ injectable + log_cum_sponsors + log_listings + log_age"
         " + generic_only + C(year)")
    r = smf.glm(f, data=pn, family=sm.families.Binomial(sm.families.links.CLogLog())
                ).fit(cov_type="cluster", cov_kwds={"groups": pn.mkt})
    keep = ["injectable", "log_cum_sponsors", "log_listings", "log_age", "generic_only"]
    print("\ncomplementary log-log hazard, year FE, SE clustered by market")
    print(tidy(r, keep, "cloglog").to_string())
    for k in keep:
        if k in r.params.index:
            lo, hi = r.conf_int().loc[k]
            print("  %-18s HR %6.2f  [%.2f, %.2f]  p=%.4f"
                  % (k, np.exp(r.params[k]), np.exp(lo), np.exp(hi), r.pvalues[k]))

    # survivorship check: re-fit on the recent window only, where a shortage
    # that started has had little time to resolve and drop out of the file
    rec = pn[pn.year >= 2022]
    r2 = smf.glm(f, data=rec, family=sm.families.Binomial(sm.families.links.CLogLog())
                 ).fit(cov_type="cluster", cov_kwds={"groups": rec.mkt})
    print("\nrestricted to 2022-2026 (onsets=%d) -- less survival filtering:"
          % rec.event.sum())
    print(tidy(r2, keep, "2022-2026").to_string())
    return r, r2


# ---------------------------------------------------------------------------
# 3. duration under length-biased sampling
# ---------------------------------------------------------------------------

def duration(m):
    print("\n" + "=" * 78)
    print("3. DURATION of active shortages")
    print("=" * 78)
    d = m[m.in_shortage == 1].dropna(subset=["elapsed_years"]).copy()
    print("active shortage markets with an onset date: %d" % len(d))
    print("\nelapsed time since onset, in years (ALL still ongoing = right-censored):")
    print(d.elapsed_years.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    print("\nby form class:")
    print(d.groupby("form_class").elapsed_years.agg(["size", "median", "mean"]).round(2).to_string())

    # Length-bias correction. A snapshot of in-progress spells over-samples long
    # ones in proportion to their length. Under stationary onset intensity the
    # observed backward-recurrence density is g(a) = S(a) / E[T], so the
    # unbiased survival function is recovered by reweighting each observed
    # spell by 1/elapsed and the mean duration by the harmonic relation below.
    a = d.elapsed_years.values
    a = a[a > 0]
    n = len(a)
    naive_mean = a.mean()
    # E[T] = 1 / E[1/A] under length-biased (backward recurrence) sampling
    ehat = n / np.sum(1.0 / a)
    w = (1.0 / a) / np.sum(1.0 / a)
    med_idx = np.argsort(a)
    cw = np.cumsum(w[med_idx])
    corrected_median = a[med_idx][np.searchsorted(cw, 0.5)]
    print("\n--- length-bias correction (assumes stationary onset intensity) ---")
    print("  naive mean elapsed (biased UP):      %6.2f y" % naive_mean)
    print("  naive median elapsed:                %6.2f y" % np.median(a))
    print("  bias-corrected mean duration:        %6.2f y" % ehat)
    print("  bias-corrected median duration:      %6.2f y" % corrected_median)
    print("  ratio naive/corrected mean:          %6.2f x" % (naive_mean / ehat))

    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(d.elapsed_years, event_observed=np.zeros(len(d)))
    print("\n  (Kaplan-Meier is degenerate here: 0 of %d spells are observed to"
          "\n   end, so S(t)=1 throughout. Reported for completeness only.)" % len(d))
    return d, ehat, naive_mean


# ---------------------------------------------------------------------------
# 4. the manufacturer-exit question
# ---------------------------------------------------------------------------

def exit_test(m):
    print("\n" + "=" * 78)
    print("4. DOES A MANUFACTURER EXIT RAISE RISK FOR THE REMAINING PRODUCTS?")
    print("=" * 78)
    s = pd.read_csv(os.path.join(DERIVED, "shortage_records.csv"),
                    parse_dates=["initial_posting", "discontinued_date"])
    s["mkt"] = s.market_ingredient + " | " + s.market_form
    tbd = s[s.discontinued_date.notna()]
    print("dated discontinuation notices: %d, in %d markets, %d firms"
          % (len(tbd), tbd.mkt.nunique(), tbd.company_norm.nunique()))
    print("notice dates span %s to %s"
          % (tbd.discontinued_date.min().date(), tbd.discontinued_date.max().date()))
    print("\nnotices by year:")
    print(tbd.discontinued_date.dt.year.value_counts().sort_index().to_string())

    post = (SNAPSHOT - tbd.discontinued_date).dt.days / 365.25
    print("\nfollow-up available after each notice (years):")
    print(post.describe(percentiles=[.5, .9]).round(2).to_string())
    print("notices with >=1y of follow-up: %d" % (post >= 1).sum())

    print("\n--> Not identified as an event study. Every dated exit lands in the"
          "\n    last 18 months of the snapshot, so there is no post-period in"
          "\n    which to observe a subsequent shortage onset. Reporting the"
          "\n    cross-sectional association instead, direction unresolved.")

    ct = pd.crosstab(m.tbd_notice, m.in_shortage)
    print("\ncross-tab of markets (rows: had a discontinuation notice):")
    print(ct.to_string())
    rate1 = m[m.tbd_notice == 1].in_shortage.mean()
    rate0 = m[m.tbd_notice == 0].in_shortage.mean()
    print("shortage rate | notice: %.1f%%   | no notice: %.1f%%   ratio %.1fx"
          % (100 * rate1, 100 * rate0, rate1 / rate0))

    r = smf.glm("in_shortage ~ tbd_notice + injectable + log_labelers +"
                " log_listings + log_age + generic_only",
                data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    print("\nconditional on the same controls:")
    print(tidy(r, ["tbd_notice", "injectable", "log_labelers"], "with notice").to_string())
    lo, hi = r.conf_int().loc["tbd_notice"]
    print("  tbd_notice OR %.2f [%.2f, %.2f] p=%.4f"
          % (np.exp(r.params.tbd_notice), np.exp(lo), np.exp(hi), r.pvalues.tbd_notice))

    # how often is the shortage literally caused by the discontinuation? if the
    # FDA-coded reason is discontinuation, exit and shortage are the same event
    cur = s[s.status == "Current"].dropna(subset=["shortage_reason"])
    rr = cur.shortage_reason.value_counts(normalize=True)
    print("\nFDA-coded reasons for currently active shortages:")
    print((100 * rr).round(1).to_string())
    return tbd


if __name__ == "__main__":
    m = load()
    tab, models = cross_section(m)
    hazard(m)
    duration(m)
    exit_test(m)
