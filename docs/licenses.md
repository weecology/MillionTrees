# Filtering by upstream license

MillionTrees aggregates dozens of independently published datasets, and they do not share
one license. Most are CC-BY 4.0, but a handful are NonCommercial, one forbids derivative
works, one is copyleft, and a few are public domain. Whether you may train on a given
annotation depends on which of those it came from.

Rather than ask you to reconstruct that mapping from the source table, every loader takes a
`licenses=` argument that keeps only the annotations whose upstream license permits your
intended use:

```python
from milliontrees import get_dataset

# Everything you can use in a commercial product.
dataset = get_dataset("TreeBoxes", root_dir="data", licenses="commercial")

# Only the most permissive terms: commercial use, derivatives, no share-alike, no copyleft.
dataset = get_dataset("TreePolygons", root_dir="data", licenses="permissive")

# Or name the licenses yourself.
dataset = get_dataset("TreePoints", root_dir="data",
                      licenses=["CC-BY-4.0", "CC0-1.0"])
```

## What filtering does and does not change

It **drops annotation rows**. It does not re-split anything. An image whose annotations are
all dropped simply leaves the dataset; every image that remains keeps the
train / validation / test assignment it already had. A license-restricted run is therefore
scored on a subset of the same benchmark splits, not on a new split, and per-source numbers
stay comparable to the published leaderboard for the sources that survive the filter.

Because sources are dropped whole, a restricted run trains on less data and usually on fewer
biomes. Report which selection you used alongside your scores.

## Selections

`licenses=` accepts a preset name, a license id, an fnmatch pattern over ids
(`"CC-BY-*"`), or a list of any of those — a list selects the **union** of what its entries
name. `licenses=None` (the default) applies no filter at all.

| Preset | Selects |
| --- | --- |
| `all` | every license, including `unknown` — the same as no filter |
| `known` | every license whose terms are confirmed |
| `commercial` | commercial use permitted (excludes the NonCommercial licenses) |
| `noncommercial` | only the NonCommercial licenses |
| `derivatives` | derivative works permitted (excludes NoDerivatives) |
| `no-share-alike` | no obligation to relicense derivatives under the same terms |
| `no-copyleft` | the license does not reach the model or code built from the data |
| `permissive` | commercial **and** derivatives **and** no share-alike **and** no copyleft |
| `public-domain` | CC0 only — no attribution obligation |
| `unknown` | only the sources whose upstream terms are unconfirmed |

Licenses currently present in a release:

| License | Commercial | Derivatives | Share-alike | Attribution | Copyleft |
| --- | --- | --- | --- | --- | --- |
| `CC0-1.0` | yes | yes | no | no | no |
| `CC-BY-3.0` | yes | yes | no | yes | no |
| `CC-BY-4.0` | yes | yes | no | yes | no |
| `CC-BY-NC-4.0` | no | yes | no | yes | no |
| `CC-BY-NC-SA-4.0` | no | yes | yes | yes | no |
| `CC-BY-NC-ND-3.0` | no | no | no | yes | no |
| `CDLA-Permissive-1.0` | yes | yes | no | yes | no |
| `AGPL-3.0-or-later` | yes | yes | yes | yes | yes |
| `unknown` | — | — | — | — | — |

These flags are a filtering aid, not legal advice. Read the license before you rely on it.

## Unconfirmed sources

A source whose upstream terms we have not confirmed is recorded as `unknown`, and **no
preset except `all` and `unknown` selects it**. Asking for `commercial` therefore drops it
rather than assuming it is safe to use. To include those sources deliberately, ask for them:

```python
get_dataset("TreeBoxes", root_dir="data", licenses=["commercial", "unknown"])
```

As of v0.24 three sources are `unknown`:

| Source | Geometry | Annotations | Why |
| --- | --- | --- | --- |
| World Resources Institute | Boxes | 29,326 | not in the manuscript source table |
| Safonova et al. 2021 | Polygons | 316 | not in the manuscript source table |
| Bohlman et al. 2008 | Polygons | 3,508 | BCI Panama; does not match the one manuscript row with a blank license |

Two further sources are mapped to a manuscript row by inference rather than by a matching
name, and are recorded as CC-BY 4.0 on that basis: `Frey et al. 2026` (the EcoSense
TLS plot, manuscript row "Liu et al. 2024") and `OSBS megaplot 2025` (manuscript row
"Johnson et al. 2021"). The reasoning is in the `notes` column of `sources.csv`.

## Inspecting what you have

Every loaded dataset exposes the licenses of the sources it is holding:

```python
dataset.source_licenses   # {'Cloutier et al. 2023': ['CC-BY-4.0'], ...}
dataset.licenses          # the selection you asked for, or None
```

To audit a packaged release before loading it:

```bash
python -m milliontrees.common.licenses                       # the license menu
python -m milliontrees.common.licenses \
    --csv data/TreeBoxes_v0.24/within-distribution.csv       # per-license row counts
python -m milliontrees.common.licenses \
    --csv data/TreeBoxes_v0.24/within-distribution.csv --licenses commercial
```

## Sources that mix licenses

Some sources are not uniform. Open Forest Observatory's field plots are contributed by
different data owners: some plots are CC-BY-NC-SA, others CC0. Two version-controlled tables
in `src/milliontrees/common/license_data/` handle this:

* `sources.csv` — one license per packaged `source` name. A mixed source records the **most
  restrictive** license it spans, so that a filter never includes a row by default that it
  should have excluded.
* `overrides.csv` — per-row refinements, matched on `source` plus an fnmatch pattern over
  `filename`. OFO's packaged filenames carry the mission id
  (`000018_ortho_616_OFO_field_2025.png`), so one row per mission assigns that plot's
  license:

  ```csv
  source,filename_pattern,license,notes
  OFO field 2025,000018_*,CC0-1.0,plot contributed under CC0
  ```

  Later rows override earlier ones, and anything no rule matches keeps the source-level
  license.

**The OFO mission rows are not populated yet.** OFO's per-plot license metadata is not
carried in the data the release is built from — the mission metadata we hold records the
license of the *imagery*, not of the field plot. Until those rows are added, every
`OFO field 2025` annotation takes the source-level `CC-BY-NC-SA-4.0`, which means a
`commercial`, `permissive` or `public-domain` selection **drops the whole source**, all 237
missions, rather than including plots it may not be entitled to. If you need the CC0 plots,
add their missions to `overrides.csv`; no code change is required.

The same mechanism covers any future source that needs sub-source granularity: add rows to
`overrides.csv`, no loader change needed.
