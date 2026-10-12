# Data

This repository does not carry its raw inputs. The files listed below were present locally when the results in the README were produced, and were deleted on 2026-10-12 to reclaim **390.69 MB** of disk. Every one of them is a download from a public upstream source, so nothing unique was lost.

This file exists so that a re-download can be *checked* rather than assumed to match. It records the exact inventory, not just the source.

Upstream sources, with links and licences, are in the README under **The data problem, stated plainly**.

## What was removed

390.69 MB in 5 files.

| Path | Size | Files |
|---|---|---|
| `data/raw` | 390.69 MB | 5 |

## Restoring it

openFDA drug shortage and enforcement endpoints. Public API, no key needed at this volume.

## Inventory

`data_manifest.csv` carries every file with its exact byte length and a fingerprint. Files of 32 MiB or less have a full `sha256`. Larger files have `headtail256`: a sha256 over the first mebibyte, the last mebibyte and the exact byte length, which catches a truncated or altered download without rereading gigabytes.

| File | Size | Fingerprint |
|---|---|---|
| `data/raw/ndc.json` | 234.05 MB | `headtail256:60eab4613a275a926113b3123d809db7b0d9e867d97d7f2be538bf2883bcd1ad` |
| `data/raw/drugsfda.json` | 118.95 MB | `headtail256:7f821adeee1f5757af75b3097fac19292586c0677a8dbd07d7e67112d17fcee8` |
| `data/raw/ndc.zip` | 26.77 MB | `sha256:ff8d370904dbb9af3e6022fc12f6add0926a76db01dc3045ed1a6cf325b91ad6` |
| `data/raw/drugsfda.zip` | 8.94 MB | `sha256:f4fbcb3840a7d820c0e6f7d0854ed456a3d90cacb14c084303422cb8ee0ef341` |
| `data/raw/shortages.json` | 1.99 MB | `sha256:b9bcf480e858c989051d516deeb354cb32fcd5b4e7e30cc70624bf924a5c356d` |

A source that is live rather than a fixed release will not reproduce these fingerprints. Where that applies it is said above.

