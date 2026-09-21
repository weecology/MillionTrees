# Filtering by upstream license

MillionTrees aggregates dozens of independently published datasets, and they do not share
one license. Most are CC-BY 4.0, but a handful are NonCommercial, some are ShareAlike, one
forbids derivative works, one is copyleft, and a few are public domain. Whether you may
train on a given annotation depends on which of those it came from.

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
| `CC-BY-SA-4.0` | yes | yes | yes | yes | no |
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

Some sources are not uniform. Two version-controlled tables in
`src/milliontrees/common/license_data/` handle this:

* `sources.csv` — one license per packaged `source` name. A mixed source records the **most
  restrictive** license it spans, so that a filter never includes a row by default that it
  should have excluded.
* `overrides.csv` — per-row refinements, matched on `source` plus an fnmatch pattern over
  `filename`. Later rows override earlier ones, and anything no rule matches keeps the
  source-level license.

A release may also carry a **`license` column**, and where it is filled it wins over both
tables. That is the only way to describe an image whose own annotations differ, since an
fnmatch rule over `filename` cannot split a single image. Rows that leave it empty fall back
to the tables, so a release where only one source fills it behaves exactly as before
everywhere else.

### Open Forest Observatory

`OFO field 2025` is the source these tables exist for, and the only one that currently needs
them. It is the one place in MillionTrees where the image and the annotation come from
**different upstream datasets with different owners**:

* the image tile is cut from an [Open Forest Observatory](https://openforestobservatory.org)
  drone mission orthomosaic — every mission in the OFO catalog is CC-BY 4.0;
* the points are a contributor's ground-reference stem map, and those plots span CC-BY 4.0,
  CC-BY-SA 4.0, CC-BY-NC-SA 4.0, CC0 1.0 and U.S. Forest Service public domain.

Using a packaged row means using both, so the license recorded for it is the **most
restrictive of the two**:

| Ground-reference plot | + CC-BY 4.0 imagery | Tiles |
| --- | --- | --- |
| CC-BY 4.0 | `CC-BY-4.0` | 2,677 |
| U.S. Forest Service public domain | `CC-BY-4.0` | 2,583 |
| CC0 1.0 | `CC-BY-4.0` | 146 |
| CC-BY-SA 4.0 | `CC-BY-SA-4.0` | 1,426 |
| CC-BY-NC-SA 4.0 | `CC-BY-NC-SA-4.0` | 968 |

The CC0 and public-domain plots therefore still carry an attribution obligation — it comes
from the imagery, not from the stem map. Attribute the plot's contributor, not OFO; OFO asks
to be credited as the mechanism through which the data were found.

The packaged filename is `{mission_id}_ortho_{tile}_OFO_field_2025.png`, so most missions need
only one rule. Seven missions cover plots under two different licenses, and their tiles are
split rule by rule:

```text
source,filename_pattern,license,notes
OFO field 2025,000091_ortho_*_OFO_field_2025.png,CC-BY-SA-4.0,"mission 000091, 15 tiles; ..."
OFO field 2025,000091_ortho_340_OFO_field_2025.png,CC-BY-NC-SA-4.0,"plot 0040 (CC BY NC SA 4.0) ..."
```

#### Tiles that straddle a plot boundary

Four tiles — `000133_ortho_589`, `000133_ortho_590`, `000136_ortho_297` and
`000136_ortho_298` — carry stems from two adjacent plots at once: plot 0049 (Vibrant Planet,
CC-BY-NC-SA 4.0) and plot 0110 (UC Davis / TNC, CC-BY-SA 4.0). The plots are 65 m apart and
share no stems, so these are genuinely two surveys of neighbouring ground, not one survey
counted twice.

A filename-keyed rule cannot split one image, so in `overrides.csv` those four tiles take the
more restrictive CC-BY-NC-SA 4.0 for all 51 of their stems. That is safe but pessimistic: it
withholds 9 CC-BY-SA annotations from a commercial-use run. The `license` column resolves them
exactly — `data_prep/process_ofo_field.py` stamps every annotation with its own plot's license
as it tiles, so a release built with it keeps all four images and drops only the 42 stems a
commercial user may not have. Until TreePoints is next rebuilt, the conservative rules apply.

#### Regenerating

`overrides.csv` is generated, not hand-edited. `data_prep/ofo_license_overrides.py` reads the
plot licenses from [Zenodo record 22731179](https://zenodo.org/records/22731179)
(`ofo_ground-reference_plots.gpkg`, column `license_short`), rebuilds the tiling from each
orthomosaic's georeferencing to work out which plot each tile came from, and refuses to write
the table unless the rebuilt tile set reproduces the packaged release exactly:

```bash
python data_prep/ofo_license_overrides.py \
    --validate /orange/ewhite/web/public/MillionTrees/TreePoints_v0.25
```

Rerun it whenever the OFO catalog is updated or the OFO tiling changes.

`Young et al. 2025 unsupervised` and `Young et al. 2025 weak supervised` are also OFO, but
they need no overrides: they are drone imagery with machine-derived labels, so CC-BY 4.0
covers them end to end.

The same mechanism covers any future source that needs sub-source granularity: add rows to
`overrides.csv`, no loader change needed.
