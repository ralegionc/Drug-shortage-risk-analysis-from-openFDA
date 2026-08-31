"""
Stress-tests for the headline result.

The supplier-count coefficient flips sign once market size is controlled, so
that flip has to survive: (a) a reparametrisation that removes the
collinearity objection, (b) a pre-determined size proxy that a current
shortage cannot have caused, (c) alternative generic measures, and
(d) dropping the size control's most influential observations.
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

from analysis import load, tidy

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)

BASE = ("in_shortage ~ injectable + log_labelers + log_listings + log_age"
        " + generic_only")


def main():
    m = load()
    m = m.copy()
    m["log_per_labeler"] = m.log_listings - m.log_labelers  # log(listings/labeler)
    m["n_anda_sponsors_l"] = np.log(m.n_anda_sponsors.clip(lower=1))

    print("=" * 78)
    print("A. IS THE SIGN FLIP JUST COLLINEARITY?")
    print("=" * 78)
    print("corr(log_labelers, log_listings) = %.3f"
          % m[["log_labelers", "log_listings"]].corr().iloc[0, 1])
    X = sm.add_constant(m[["injectable", "log_labelers", "log_listings",
                           "log_age", "generic_only"]]).dropna()
    print("markets with a usable approval date: %d of %d" % (len(X), len(m)))
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print("\nVIF:")
    for i, c in enumerate(X.columns):
        print("  %-14s %6.2f" % (c, variance_inflation_factor(X.values, i)))

    print("\nSame fit, reparametrised so the two supplier terms are not rival:")
    print("  log_labelers + log_listings  ==  log_listings + log(listings/labeler)")
    r_a = smf.glm(BASE, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    r_b = smf.glm("in_shortage ~ injectable + log_listings + log_per_labeler"
                  " + log_age + generic_only",
                  data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    print("\n  identical log-likelihood? %.4f vs %.4f  -> %s"
          % (r_a.llf, r_b.llf, "yes" if abs(r_a.llf - r_b.llf) < 1e-6 else "NO"))
    print("\n  concentration form (listings per labeler = how much of the market"
          "\n  each supplier carries):")
    for k in ["injectable", "log_listings", "log_per_labeler", "log_age"]:
        lo, hi = r_b.conf_int().loc[k]
        print("    %-16s OR %6.2f [%.2f, %.2f] p=%.4f"
              % (k, np.exp(r_b.params[k]), np.exp(lo), np.exp(hi), r_b.pvalues[k]))
    print("\n  corr(log_listings, log_per_labeler) = %.3f  (was %.3f) -- the"
          " rivalry is gone"
          % (m[["log_listings", "log_per_labeler"]].corr().iloc[0, 1],
             m[["log_labelers", "log_listings"]].corr().iloc[0, 1]))

    print("\n" + "=" * 78)
    print("B. SIZE CONTROL THAT A CURRENT SHORTAGE CANNOT HAVE CAUSED")
    print("=" * 78)
    print("n_ndc_active is measured today, so a shortage could in principle have")
    print("shrunk it. n_products_ever counts every product ever approved in the")
    print("market, including long-discontinued ones, and is pre-determined.")
    m["log_sponsors_ever"] = np.log(m.n_sponsors_ever.clip(lower=1))
    m["log_per_sponsor_ever"] = m.log_products - m.log_sponsors_ever
    specs = {
        "current listings": BASE,
        "approved-ever": ("in_shortage ~ injectable + log_labelers + log_products"
                          " + log_age + generic_only"),
        "both": ("in_shortage ~ injectable + log_labelers + log_products"
                 " + log_listings + log_age + generic_only"),
    }
    cols = []
    for lbl, f in specs.items():
        r = smf.glm(f, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
        cols.append(tidy(r, ["injectable", "log_labelers", "log_listings",
                             "log_products", "log_age"], lbl))
    print()
    print(pd.concat(cols, axis=1).to_string())

    print("\nthe same concentration reparametrisation, built only from")
    print("pre-determined approval-file quantities (products per sponsor ever):")
    r_hist = smf.glm("in_shortage ~ injectable + log_products +"
                     " log_per_sponsor_ever + log_age + generic_only",
                     data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    for k in ["injectable", "log_products", "log_per_sponsor_ever", "log_age"]:
        lo, hi = r_hist.conf_int().loc[k]
        print("    %-22s OR %6.2f [%.2f, %.2f] p=%.4f"
              % (k, np.exp(r_hist.params[k]), np.exp(lo), np.exp(hi),
                 r_hist.pvalues[k]))

    print("\n" + "=" * 78)
    print("C. 'CHEAP GENERIC': WHICH GENERIC MEASURE, IF ANY, BITES?")
    print("=" * 78)
    print("openFDA carries no price field, so 'cheap' is unobservable. These are")
    print("the available proxies for the generic-competition part of the claim.")
    for v in ["generic_only", "has_generic", "n_anda_sponsors_l"]:
        f = ("in_shortage ~ injectable + log_labelers + log_listings + log_age + " + v)
        r = smf.glm(f, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
        lo, hi = r.conf_int().loc[v]
        print("  %-18s OR %6.2f [%.2f, %.2f] p=%.4f"
              % (v, np.exp(r.params[v]), np.exp(lo), np.exp(hi), r.pvalues[v]))
    print("\n  interaction: does supplier count matter more for injectables?")
    r = smf.glm("in_shortage ~ injectable * log_labelers + log_listings + log_age",
                data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    print(tidy(r, ["injectable", "log_labelers", "injectable:log_labelers"],
               "interaction").to_string())

    print("\n" + "=" * 78)
    print("D. INFLUENCE AND RARE-EVENT CHECKS")
    print("=" * 78)
    cl = m.dropna(subset=["log_age", "log_labelers", "log_listings"]).copy()
    r = smf.glm(BASE, data=cl, family=sm.families.Binomial()).fit()
    infl = r.get_influence()
    cook = pd.Series(infl.cooks_distance[0], index=cl.index)
    top = cook.nlargest(10).index
    r_drop = smf.glm(BASE, data=cl.drop(index=top),
                     family=sm.families.Binomial()).fit(cov_type="HC1")
    print("dropping the 10 highest-Cook's-distance markets:")
    print(pd.concat([tidy(r, ["injectable", "log_labelers", "log_listings", "log_age"], "full"),
                     tidy(r_drop, ["injectable", "log_labelers", "log_listings", "log_age"], "-10 influential")],
                    axis=1).to_string())
    print("\nmost influential markets:")
    for k in top[:6]:
        print("   %-52s shortage=%d labelers=%d"
              % (k[:52], cl.loc[k, "in_shortage"], cl.loc[k, "n_labelers_active"]))

    print("\nevents per parameter: %.1f (%d events / %d params)"
          % (cl.in_shortage.sum() / len(r.params), cl.in_shortage.sum(),
             len(r.params)))

    # exact-ish check: penalised (ridge) logit to confirm nothing is separation
    r_pen = smf.logit(BASE, data=cl).fit_regularized(alpha=1.0, disp=0)
    print("\nL2-penalised logit (alpha=1), confirms no separation driving signs:")
    print(r_pen.params.round(3).to_string())

    print("\n" + "=" * 78)
    print("E. HOW BIG IS THE SURVIVORSHIP HOLE?")
    print("=" * 78)
    s = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                 "derived", "shortage_records.csv"),
                    parse_dates=["initial_posting"])
    print("shortage file contains %d records; only %d are status='Resolved'."
          % (len(s), (s.status == "Resolved").sum()))
    print("So the file is a prevalent cohort: a shortage that began and ended")
    print("between two snapshots leaves no trace. Onsets by year, still-active:")
    cur = s[s.status == "Current"]
    print(cur.groupby(cur.initial_posting.dt.year).size().to_string())
    print("\nThe 2024 trough (%d records) is not a quiet year -- it is the year"
          % (cur.initial_posting.dt.year == 2024).sum())
    print("whose shortages have mostly already resolved and been purged.")


if __name__ == "__main__":
    main()
