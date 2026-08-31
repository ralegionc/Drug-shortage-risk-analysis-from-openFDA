"""
Append a dated snapshot of the openFDA shortage file to a local store.

This is the fix for the central data problem. /drug/shortages.json reports
current state: as of 2026-08-22 it held 1,628 records of which exactly 10 were
status='Resolved'. A shortage that begins and ends between two observations
leaves no trace at all, so the live endpoint can never support an incidence or
duration model on its own -- it is a prevalent cohort, and prevalent cohorts
over-represent long spells.

Running this on a schedule turns that into a genuine longitudinal panel: each
NDC's first and last appearance bracket its shortage spell, and disappearance
between two snapshots is an observed resolution rather than a silent deletion.
After ~12 monthly runs the duration model in analysis.py can be re-estimated
without the stationarity assumption the length-bias correction currently needs.

Usage:
    python snapshot.py            # capture today
    python snapshot.py --report   # summarise what has accumulated so far
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.request

import pandas as pd

STORE = os.path.join(os.path.dirname(__file__), "..", "data", "snapshots")
os.makedirs(STORE, exist_ok=True)

ENDPOINT = "https://api.fda.gov/drug/shortages.json"
PAGE = 1000

# fields worth keeping per record; the rest are contact details and prose
FIELDS = ["package_ndc", "generic_name", "company_name", "dosage_form",
          "status", "availability", "update_type", "shortage_reason",
          "initial_posting_date", "update_date", "discontinued_date",
          "change_date", "therapeutic_category", "presentation"]


def fetch_all():
    out, skip = [], 0
    while True:
        url = "%s?limit=%d&skip=%d" % (ENDPOINT, PAGE, skip)
        with urllib.request.urlopen(url) as fh:
            d = json.load(fh)
        rows = d["results"]
        out.extend(rows)
        total = d["meta"]["results"]["total"]
        last_updated = d["meta"]["last_updated"]
        skip += PAGE
        if skip >= total or not rows:
            return out, total, last_updated
        time.sleep(0.5)


def capture():
    rows, total, last_updated = fetch_all()
    today = dt.date.today().isoformat()
    recs = []
    for r in rows:
        rec = {k: r.get(k) for k in FIELDS}
        tc = r.get("therapeutic_category")
        rec["therapeutic_category"] = "; ".join(tc) if isinstance(tc, list) else tc
        ofd = r.get("openfda") or {}
        rec["application_number"] = (ofd.get("application_number") or [None])[0]
        rec["snapshot_date"] = today
        rec["endpoint_last_updated"] = last_updated
        recs.append(rec)
    df = pd.DataFrame(recs)
    path = os.path.join(STORE, "shortages_%s.csv" % today)
    df.to_csv(path, index=False)
    print("captured %d records (endpoint total %d, last_updated %s)"
          % (len(df), total, last_updated))
    print("wrote %s" % path)
    return df


def report():
    files = sorted(f for f in os.listdir(STORE) if f.startswith("shortages_"))
    if not files:
        print("no snapshots yet -- run `python snapshot.py` first")
        return
    frames = [pd.read_csv(os.path.join(STORE, f)) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df.snapshot_date)
    print("snapshots: %d, from %s to %s"
          % (len(files), df.snapshot_date.min().date(), df.snapshot_date.max().date()))

    if len(files) < 2:
        print("\nOne snapshot only. Resolutions become observable from the")
        print("second capture onward: an NDC present in snapshot t and absent")
        print("in t+1 has left the shortage list, which dates the spell end.")
        return

    seen = df.groupby("package_ndc").agg(
        first_seen=("snapshot_date", "min"),
        last_seen=("snapshot_date", "max"),
        n_snapshots=("snapshot_date", "nunique"))
    latest = df.snapshot_date.max()
    seen["resolved"] = seen.last_seen < latest
    seen["observed_days"] = (seen.last_seen - seen.first_seen).dt.days
    print("\ndistinct NDCs tracked: %d" % len(seen))
    print("left the list at some point (observed resolutions): %d"
          % seen.resolved.sum())
    if seen.resolved.any():
        r = seen[seen.resolved]
        print("\nobserved spell length in days (right-truncated by snapshot spacing):")
        print(r.observed_days.describe(percentiles=[.25, .5, .75]).round(1).to_string())
    out = os.path.join(STORE, "..", "derived", "ndc_spells.csv")
    seen.to_csv(out)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="summarise accumulated snapshots instead of capturing")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        capture()
        report()
