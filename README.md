# What puts a drug into shortage

Panel analysis of US drug shortages built from three unauthenticated openFDA bulk
endpoints, captured 2026-08-22.

**Report:** `output/report.html`

## Headline

| Claim | Status | Estimate |
|---|---|---|
| Sterile injectables carry far higher shortage risk | **identified** | OR 9.88 (4.69–20.78), AME +6.8 pp |
| Supplier concentration raises risk | **partly identified** | OR 2.45 (1.40–4.30) per doubling of market carried per supplier — but only with current-listings size control |
| "Cheap generic" adds explanatory power | **not supported** | No generic proxy significant; price absent from openFDA entirely |
| A manufacturer exit raises risk for remaining products | **not identified** | All 440 dated exits fall in the last 18 months; median follow-up 0.5 y |

## The data problem, stated plainly

`/drug/shortages.json` returns current state. Of 1,628 records, **10** carry status
`Resolved`, eight of which duplicate a single hydrocortisone entry. The file is a
**prevalent cohort**: it samples shortages in progress, not shortages that occurred.
A shortage that began in 2019 and ended in 2020 leaves no trace.

Two related gaps close off the obvious workarounds:

- The NDC directory has **zero past marketing-end dates** — delisted products are purged,
  so market exits cannot be dated from it.
- `drugsfda` retains 23,496 discontinued products but attaches **no date** to
  discontinuation.

I tested using a firm's last approved submission as an exit-date proxy against the 294
discontinuations that carry a true date: it leads the real date 90% of the time but with a
median lead of 3.0 years (IQR 1.3–5.3). Too coarse for an annual event study, so it is not
used.

Consequences, and how the analysis responds:

| Consequence | Response |
|---|---|
| Shortage *incidence* is unobservable | Primary estimand is cross-sectional prevalence, which the snapshot identifies exactly |
| Onset hazard entangles onset with duration | Reported but not headlined; injectable HR falls 10.75 → 3.99 restricting to 2022–2026 |
| Elapsed durations are length-biased upward | Backward-recurrence correction: `E[T] = 1/E[1/A]` gives 2.55 y against a naive 5.64 y |
| Exits have no post-period | Reported as not identified; `snapshot.py` ships to fix it going forward |

## The central modelling result

Raw, more suppliers correlates with *more* shortage (injectables: 0.8% at one supplier,
15.7% at nine or more) because supplier count proxies for market size, which openFDA never
reports. Adding a size control flips the coefficient from OR 1.90 to **OR 0.41**.

Because the two supplier terms correlate at 0.94, the flip is restated as an identical-fit
reparametrisation — market size plus **listings per supplier** — which correlates at only
0.68 and gives OR 2.45 (1.40–4.30).

**Where it is fragile:** rebuilt from pre-determined approval-file quantities (products ever
approved per sponsor) the same measure gives OR 1.20 (0.72–2.00), p = 0.47. The result rests
on judging current listings the better proxy for present-day demand. A reader who disagrees
should treat concentration as unresolved.

## Pipeline

```bash
pip install pandas numpy statsmodels lifelines
cd src
python build_universe.py    # parse drugsfda / ndc / shortages -> products, listings, records
python build_panel.py       # markets cross-section + market-year panel
python analysis.py          # cross-section, hazard, duration, exit
python robustness.py        # collinearity, alternative size control, influence, separation
python export_figures.py    # output/figures.json, drives the report charts
```

Raw bulk files are not committed (≈370 MB extracted). Re-fetch:

```bash
curl -o data/raw/drugsfda.zip https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip
```

## Ongoing capture — the fix for the prevalent-cohort problem

```bash
python src/snapshot.py            # append a dated capture
python src/snapshot.py --report   # summarise what has accumulated
```

Each run writes `data/snapshots/shortages_<date>.csv`. An NDC present at *t* and absent at
*t+1* is an **observed resolution with a date attached**. After roughly twelve monthly runs
the duration model can be re-estimated without the stationarity assumption the length-bias
correction currently needs, and exits acquire a real post-period — at which point the
natural experiment in the brief becomes answerable.

## Definitions

- **Market** — normalised active-ingredient set × dosage-form class (7 classes, route-aware).
  4,111 total; 2,149 marketed. This is the level at which substitution happens: a hospital
  short of one firm's carboplatin vial substitutes another firm's vial, not tablets.
- **Marketed** — ≥1 non-discontinued drugsfda product *and* ≥1 active NDC listing.
- **Outcome** — ≥1 shortage record with status `Current` at the snapshot.
- **Linkage** — ingredient+form exact match → openFDA application number → unique-form match.
  97.6% of records linked; the 39 unlinked (mostly parenteral-nutrition amino-acid
  combinations) are excluded rather than assigned.

## Caveats

- No price, volume, or revenue data exists in openFDA. "Cheap" is untestable here.
- Drug-class controls use NDC `pharm_class_epc`, populated for ~38% of listings; an explicit
  `UNKNOWN` level carries the rest.
- 62 events across 2,149 markets — about 10 events per parameter. Adequate for the reported
  specifications, not for richer ones.
- The cross-sectional prevalence estimand confounds onset with duration by construction: a
  covariate that lengthens shortages raises prevalence identically to one that causes them.
