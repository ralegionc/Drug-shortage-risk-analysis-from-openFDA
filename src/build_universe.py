"""
Build the product universe and the molecule-form market panel from openFDA.

Sources (all unauthenticated, bulk endpoints):
  data/raw/drugsfda.json  - 29,277 applications / 51,660 products, approval dates, marketing status
  data/raw/ndc.json       - 137,198 currently-listed products, marketing_start_date
  data/raw/shortages.json - 1,628 shortage/discontinuation presentation records

Output:
  data/derived/products.csv  - one row per drugsfda product
  data/derived/markets.csv   - one row per molecule x form-class market
  data/derived/shortage_events.csv - one row per (market, shortage record)
"""

import json
import os
import re
import collections
import datetime as dt

import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
os.makedirs(DERIVED, exist_ok=True)

SNAPSHOT = dt.date(2026, 8, 22)  # meta.last_updated on both endpoints


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------

# salt / hydrate suffixes stripped so that e.g. "METOPROLOL TARTRATE" and
# "METOPROLOL SUCCINATE" collapse to the same moiety only when we ask them to.
# We keep salts distinct by default (they are distinct products clinically and
# for manufacturing) but strip hydration state, which is pure noise.
_HYDRATE = re.compile(
    r"\b(ANHYDROUS|MONOHYDRATE|DIHYDRATE|TRIHYDRATE|HEMIHYDRATE|"
    r"SESQUIHYDRATE|PENTAHYDRATE|HEPTAHYDRATE)\b"
)
_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_WS = re.compile(r"\s+")


def norm_ingredient(name):
    if not name:
        return ""
    s = name.upper().strip()
    s = _HYDRATE.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def ingredient_key(names):
    """Order-independent key for a combination product."""
    parts = sorted({norm_ingredient(n) for n in names if norm_ingredient(n)})
    return "|".join(parts)


# Dosage-form -> coarse class. Order matters: first match wins.
_FORM_RULES = [
    ("INJECTABLE", re.compile(r"INJECT|INFUSION|IMPLANT|PELLET")),
    ("OPHTHALMIC", re.compile(r"OPHTH")),
    ("INHALATION", re.compile(r"INHAL|AEROSOL|SPRAY, METERED|GAS")),
    ("ORAL_SOLID", re.compile(r"TABLET|CAPSULE|GRANULE|POWDER, FOR SUSPENSION|LOZENGE|FILM")),
    ("ORAL_LIQUID", re.compile(r"SOLUTION, ORAL|ORAL SOLUTION|SUSPENSION|SYRUP|ELIXIR|CONCENTRATE")),
    ("TOPICAL", re.compile(r"CREAM|OINTMENT|GEL|LOTION|PATCH|SHAMPOO|PASTE|SUPPOSITORY")),
    ("OTHER_LIQUID", re.compile(r"SOLUTION|IRRIGANT|LIQUID")),
]

# Routes that imply a sterile parenteral product even when the dosage form is
# vague (e.g. "SOLUTION" delivered intravenously).
_STERILE_ROUTE = re.compile(
    r"INTRAVENOUS|INTRAMUSCULAR|SUBCUTANEOUS|INTRATHECAL|EPIDURAL|"
    r"INTRA-ARTERIAL|INTRAARTICULAR|INTRAOCULAR|INTRAVITREAL|PARENTERAL|"
    r"INTRAPERITONEAL|INTRACAVERNOUS|INTRAPLEURAL|IRRIGATION"
)


# Route-based overrides. FDA splits the same product between dosage_form and
# route inconsistently across the three endpoints (pilocarpine ophthalmic
# solution is TOPICAL/SOLUTION in drugsfda but "Ophthalmic Solution" in the
# shortage file), so route has to be consulted before falling back to the
# dosage-form string alone.
_ROUTE_RULES = [
    ("OPHTHALMIC", re.compile(r"OPHTHALMIC|INTRAOCULAR|INTRAVITREAL")),
    ("INHALATION", re.compile(r"RESPIRATORY|INHALATION|NASAL")),
    ("TOPICAL", re.compile(r"TOPICAL|TRANSDERMAL|RECTAL|VAGINAL|CUTANEOUS")),
]


def form_class(dosage_form, route):
    df = (dosage_form or "").upper()
    rt = (route or "").upper()
    # sterile parenteral wins over everything: it is the manufacturing
    # constraint we care about, whatever the container is called
    if _STERILE_ROUTE.search(rt) or "INJECT" in df or "INFUSION" in df:
        return "INJECTABLE"
    for label, rx in _ROUTE_RULES:
        if rx.search(rt):
            return label
    for label, rx in _FORM_RULES:
        if rx.search(df):
            return label
    if "ORAL" in rt:
        return "ORAL_SOLID" if re.search(r"TABLET|CAPSULE", df) else "ORAL_LIQUID"
    return "OTHER"


def is_sterile(dosage_form, route):
    fc = form_class(dosage_form, route)
    if fc == "INJECTABLE":
        return True
    df = (dosage_form or "").upper()
    rt = (route or "").upper()
    return bool(_STERILE_ROUTE.search(rt) or "INJECT" in df)


# Sponsor names in drugsfda are applicant short-codes, already fairly canonical,
# but they carry corporate suffixes inconsistently. Strip them before counting
# distinct manufacturers so "TEVA" and "TEVA PHARMS USA" do not double-count.
_CORP = re.compile(
    r"\b(INC|LLC|LTD|CORP|CO|COMPANY|PLC|GMBH|SA|SAS|AG|NV|BV|AB|AS|"
    r"PHARM|PHARMS|PHARMA|PHARMACEUTICAL|PHARMACEUTICALS|LABS|LABORATORY|"
    r"LABORATORIES|HLDGS|HOLDING|HOLDINGS|GROUP|USA|US|AMERICA|INTL|"
    r"INTERNATIONAL|DIV|SUB|LP|LLP|PTY|SPA|SRL|KG|OY|A S)\b"
)


def norm_sponsor(name):
    if not name:
        return ""
    s = name.upper()
    s = _PUNCT.sub(" ", s)
    s = _CORP.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s or (name.upper().strip())


def parse_fda_date(s):
    if not s or len(s) < 8:
        return None
    try:
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def parse_slash_date(s):
    """Shortage endpoint uses MM/DD/YYYY."""
    if not s:
        return None
    try:
        m, d, y = s.split("/")
        return dt.date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------
# 1. products from drugsfda
# --------------------------------------------------------------------------

def build_products():
    apps = json.load(open(os.path.join(RAW, "drugsfda.json")))["results"]
    rows = []
    for a in apps:
        appno = a.get("application_number", "")
        apptype = appno[:3] if appno[:3] in ("AND", "NDA", "BLA") else "OTHER"
        apptype = "ANDA" if apptype == "AND" else apptype
        sponsor = a.get("sponsor_name", "")

        # approval date: original approval, else earliest approved submission
        orig, earliest = None, None
        for s in a.get("submissions", []):
            d = parse_fda_date(s.get("submission_status_date"))
            if d is None or s.get("submission_status") != "AP":
                continue
            if earliest is None or d < earliest:
                earliest = d
            if s.get("submission_type") == "ORIG" and (orig is None or d < orig):
                orig = d
        approval = orig or earliest

        # last approved submission of any kind: proxy for "still actively maintained"
        last_ap = None
        for s in a.get("submissions", []):
            d = parse_fda_date(s.get("submission_status_date"))
            if d is not None and s.get("submission_status") == "AP":
                if last_ap is None or d > last_ap:
                    last_ap = d

        n_sub = len(a.get("submissions", []))
        ofd = a.get("openfda", {}) or {}

        for p in a.get("products", []):
            ings = [i.get("name") for i in p.get("active_ingredients", []) or []]
            rows.append({
                "application_number": appno,
                "app_type": apptype,
                "sponsor_name": sponsor,
                "sponsor_norm": norm_sponsor(sponsor),
                "product_number": p.get("product_number"),
                "brand_name": p.get("brand_name"),
                "ingredient_key": ingredient_key(ings),
                "n_ingredients": len({norm_ingredient(i) for i in ings if i}),
                "dosage_form": p.get("dosage_form"),
                "route": p.get("route"),
                "form_class": form_class(p.get("dosage_form"), p.get("route")),
                "sterile": is_sterile(p.get("dosage_form"), p.get("route")),
                "marketing_status": p.get("marketing_status"),
                "reference_drug": p.get("reference_drug"),
                "approval_date": approval,
                "last_ap_date": last_ap,
                "n_submissions": n_sub,
                "has_ndc_in_openfda": bool(ofd.get("product_ndc")),
            })
    df = pd.DataFrame(rows)
    df = df[df["ingredient_key"] != ""].copy()
    return df


# --------------------------------------------------------------------------
# 2. active listings from the NDC directory (dates market presence today)
# --------------------------------------------------------------------------

def build_ndc():
    prods = json.load(open(os.path.join(RAW, "ndc.json")))["results"]
    rows = []
    for p in prods:
        if p.get("product_type") not in ("HUMAN PRESCRIPTION DRUG", "PLASMA DERIVATIVE"):
            continue
        ings = [i.get("name") for i in p.get("active_ingredients", []) or []]
        route = ", ".join(p.get("route") or [])
        rows.append({
            "product_ndc": p.get("product_ndc"),
            "labeler_norm": norm_sponsor(p.get("labeler_name")),
            "application_number": p.get("application_number"),
            "marketing_category": p.get("marketing_category"),
            "ingredient_key": ingredient_key(ings),
            "form_class": form_class(p.get("dosage_form"), route),
            "marketing_start": parse_fda_date(p.get("marketing_start_date")),
            "marketing_end": parse_fda_date(p.get("marketing_end_date")),
        })
    df = pd.DataFrame(rows)
    return df[df["ingredient_key"] != ""].copy()


# --------------------------------------------------------------------------
# 3. shortage records -> market-level events
# --------------------------------------------------------------------------

def build_shortages():
    recs = json.load(open(os.path.join(RAW, "shortages.json")))
    rows = []
    for r in recs:
        ofd = r.get("openfda", {}) or {}
        subs = ofd.get("substance_name") or []
        # fall back to parsing the generic_name string when openfda is absent
        if not subs:
            gn = r.get("generic_name", "")
            gn = re.sub(
                r",?\s*(TABLET|CAPSULE|INJECTION|SOLUTION|SUSPENSION|CREAM|"
                r"OINTMENT|SUPPOSITORY|IRRIGANT|CONCENTRATE|POWDER|GEL|"
                r"FILM COATED|EXTENDED RELEASE|DELAYED RELEASE|CHEWABLE|ORAL).*$",
                "", gn.upper())
            subs = [s for s in re.split(r",| AND ", gn) if s.strip()]
        route = ", ".join(ofd.get("route") or [])
        rows.append({
            "package_ndc": r.get("package_ndc"),
            "generic_name": r.get("generic_name"),
            "company_norm": norm_sponsor(r.get("company_name")),
            "ingredient_key": ingredient_key(subs),
            "dosage_form": r.get("dosage_form"),
            "form_class": form_class(r.get("dosage_form"), route),
            "status": r.get("status"),
            "availability": r.get("availability"),
            "update_type": r.get("update_type"),
            "shortage_reason": r.get("shortage_reason"),
            "therapeutic_category": "; ".join(r.get("therapeutic_category") or []),
            "initial_posting": parse_slash_date(r.get("initial_posting_date")),
            "update_date": parse_slash_date(r.get("update_date")),
            "discontinued_date": parse_slash_date(r.get("discontinued_date")),
            "change_date": parse_slash_date(r.get("change_date")),
            "application_number": (ofd.get("application_number") or [None])[0],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. resolve each shortage record onto a market in the drugsfda universe
# --------------------------------------------------------------------------

def resolve_markets(sh, products):
    """Assign every shortage record a (ingredient_key, form_class) that exists
    in the approved-product universe.

    The two files disagree on naming often enough that a single join rule
    leaves ~8% of records stranded, so we fall back in order of authority:
      1. openfda application_number -> the drugsfda application it names
      2. exact (ingredient, form) match
      3. ingredient match with a unique form in the universe
    """
    universe = set(zip(products["ingredient_key"], products["form_class"]))
    by_app = (products.groupby("application_number")[["ingredient_key", "form_class"]]
                      .agg(lambda c: c.value_counts().index[0]))
    app_market = {a: (r["ingredient_key"], r["form_class"])
                  for a, r in by_app.iterrows()}
    forms_by_ing = products.groupby("ingredient_key")["form_class"].agg(set).to_dict()

    keys, srcs = [], []
    for _, r in sh.iterrows():
        ik, fc = r["ingredient_key"], r["form_class"]
        appno = r["application_number"]
        if (ik, fc) in universe:
            keys.append((ik, fc)); srcs.append("exact")
        elif isinstance(appno, str) and appno in app_market:
            keys.append(app_market[appno]); srcs.append("application_number")
        elif ik in forms_by_ing and len(forms_by_ing[ik]) == 1:
            keys.append((ik, next(iter(forms_by_ing[ik])))); srcs.append("unique_form")
        else:
            keys.append((ik, fc)); srcs.append("unmatched")
    sh = sh.copy()
    sh["market_ingredient"] = [k[0] for k in keys]
    sh["market_form"] = [k[1] for k in keys]
    sh["link_source"] = srcs
    return sh


if __name__ == "__main__":
    print("parsing drugsfda ...", flush=True)
    products = build_products()
    products.to_csv(os.path.join(DERIVED, "products.csv"), index=False)
    print("  products:", len(products))
    print("  distinct ingredient keys:", products["ingredient_key"].nunique())
    print("  distinct sponsors:", products["sponsor_norm"].nunique())
    print("  form_class:\n", products["form_class"].value_counts().to_string())

    print("\nparsing ndc ...", flush=True)
    ndc = build_ndc()
    ndc.to_csv(os.path.join(DERIVED, "ndc_active.csv"), index=False)
    print("  active Rx listings:", len(ndc))
    print("  distinct ingredient keys:", ndc["ingredient_key"].nunique())

    print("\nparsing shortages ...", flush=True)
    sh = build_shortages()
    sh = resolve_markets(sh, products)
    sh.to_csv(os.path.join(DERIVED, "shortage_records.csv"), index=False)
    print("  records:", len(sh))
    print("  link source:\n", sh["link_source"].value_counts().to_string())
    matched = sh[sh["link_source"] != "unmatched"]
    print("  matched records: %d (%.1f%%)" % (len(matched), 100 * len(matched) / len(sh)))
    print("  distinct markets touched:",
          matched.groupby(["market_ingredient", "market_form"]).ngroups)
    print("  matched markets that are injectable:",
          matched[matched.market_form == "INJECTABLE"]
              .groupby(["market_ingredient", "market_form"]).ngroups)
