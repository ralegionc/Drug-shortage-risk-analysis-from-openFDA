"""Dump the exact numbers the report's figures are drawn from, so the charts
cannot drift from the models."""

import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

from analysis import load

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
os.makedirs(OUT, exist_ok=True)


def or_ci(res, k):
    lo, hi = res.conf_int().loc[k]
    return {"or": float(np.exp(res.params[k])), "lo": float(np.exp(lo)),
            "hi": float(np.exp(hi)), "p": float(res.pvalues[k])}


def main():
    m = load()
    m["log_per_labeler"] = m.log_listings - m.log_labelers
    m["log_sponsors_ever"] = np.log(m.n_sponsors_ever.clip(lower=1))
    m["log_per_sponsor_ever"] = m.log_products - m.log_sponsors_ever
    out = {}

    # --- 1. shortage rate by form class ---------------------------------
    t = m.groupby("form_class").agg(n=("in_shortage", "size"),
                                    k=("in_shortage", "sum"))
    t = t[t.n >= 50]
    t["rate"] = 100 * t.k / t.n
    out["by_form"] = [{"form": i.replace("_", " ").title(), "n": int(r.n),
                       "k": int(r.k), "rate": round(float(r.rate), 2)}
                      for i, r in t.sort_values("rate", ascending=False).iterrows()]

    # --- 2. the sign flip: supplier-count OR as controls are added -------
    specs = [
        ("no controls", "in_shortage ~ injectable + log_labelers"),
        ("+ market size", "in_shortage ~ injectable + log_labelers + log_listings"),
        ("+ age", "in_shortage ~ injectable + log_labelers + log_listings + log_age"),
        ("+ generic", "in_shortage ~ injectable + log_labelers + log_listings + log_age + generic_only"),
        ("+ drug-class FE", "in_shortage ~ injectable + log_labelers + log_listings + log_age + generic_only + C(epc_grp)"),
    ]
    flip = []
    for lbl, f in specs:
        r = smf.glm(f, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
        d = or_ci(r, "log_labelers"); d["spec"] = lbl
        flip.append(d)
    out["sign_flip"] = flip

    # --- 3. main forest plot (full spec) ---------------------------------
    full = ("in_shortage ~ injectable + log_labelers + log_listings + log_age"
            " + generic_only + C(epc_grp)")
    r = smf.glm(full, data=m, family=sm.families.Binomial()).fit(cov_type="HC1")
    labels = {"injectable": "Sterile injectable",
              "log_labelers": "Supplier count (log)",
              "log_listings": "Market size, listings (log)",
              "log_age": "Years since first approval (log)",
              "generic_only": "Generic-only market"}
    out["forest"] = [dict(term=labels[k], **or_ci(r, k)) for k in labels]
    out["n_markets"] = int(len(m))
    out["n_events"] = int(m.in_shortage.sum())

    # --- 4. concentration reparametrisation, both size measures ----------
    rc = smf.glm("in_shortage ~ injectable + log_listings + log_per_labeler"
                 " + log_age + generic_only", data=m,
                 family=sm.families.Binomial()).fit(cov_type="HC1")
    rh = smf.glm("in_shortage ~ injectable + log_products + log_per_sponsor_ever"
                 " + log_age + generic_only", data=m,
                 family=sm.families.Binomial()).fit(cov_type="HC1")
    out["concentration"] = [
        dict(label="Current listings per supplier", **or_ci(rc, "log_per_labeler")),
        dict(label="Products ever approved per sponsor",
             **or_ci(rh, "log_per_sponsor_ever")),
    ]

    # --- 5. duration ------------------------------------------------------
    d = m[m.in_shortage == 1].dropna(subset=["elapsed_years"])
    a = d.elapsed_years.values
    a = a[a > 0]
    ehat = len(a) / np.sum(1.0 / a)
    w = (1.0 / a) / np.sum(1.0 / a)
    o = np.argsort(a)
    cmed = float(a[o][np.searchsorted(np.cumsum(w[o]), 0.5)])
    out["duration"] = {
        "n": int(len(a)),
        "naive_mean": round(float(a.mean()), 2),
        "naive_median": round(float(np.median(a)), 2),
        "corr_mean": round(float(ehat), 2),
        "corr_median": round(cmed, 2),
        # note: g["size"], not g.size -- the latter is the Series length attribute
        "by_form": [{"form": i.replace("_", " ").title(), "n": int(g["size"]),
                     "median": round(float(g["median"]), 2)}
                    for i, g in d.groupby("form_class").elapsed_years
                                 .agg(["size", "median"]).iterrows()
                    if g["size"] >= 5],
        "hist": np.histogram(a, bins=[0, 2, 4, 6, 8, 10, 16])[0].tolist(),
    }

    # --- 6. survivorship: onsets by year ---------------------------------
    s = pd.read_csv(os.path.join(DERIVED, "shortage_records.csv"),
                    parse_dates=["initial_posting"])
    cur = s[s.status == "Current"]
    yr = cur.groupby(cur.initial_posting.dt.year).size()
    yr = yr.reindex(range(2012, 2027), fill_value=0)
    out["onsets"] = [{"year": int(y), "n": int(v)} for y, v in yr.items()]
    out["resolved_records"] = int((s.status == "Resolved").sum())
    out["total_records"] = int(len(s))

    # --- 7. reasons -------------------------------------------------------
    rr = cur.dropna(subset=["shortage_reason"]).shortage_reason.value_counts(normalize=True)
    out["reasons"] = [{"reason": k, "pct": round(100 * float(v), 1)}
                      for k, v in rr.items()]

    with open(os.path.join(OUT, "figures.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote output/figures.json")
    print(json.dumps({k: out[k] for k in
                      ["n_markets", "n_events", "duration", "concentration"]},
                     indent=1)[:1200])


if __name__ == "__main__":
    main()
