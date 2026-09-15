"""Upstream licenses: which annotations a user is allowed to train on.

Every source in a MillionTrees release arrives under its own upstream license, and those
licenses are not interchangeable -- a few are non-commercial, one forbids derivatives, one
is copyleft. A user who needs to respect those terms should not have to hand-maintain a
list of source names, so this module keeps the mapping once, next to the code that loads
the data, and turns it into a row filter the three loaders accept as ``licenses=``.

Two version-controlled tables in ``license_data/`` drive it:

* ``sources.csv`` -- one license per packaged ``source`` name. A source whose terms have
  not been confirmed is recorded as ``unknown``.
* ``overrides.csv`` -- per-row refinements for sources that mix licenses, matched on
  ``source`` + an fnmatch pattern over ``filename``. Open Forest Observatory is the case
  that needs this: its field plots come from different data owners (CC-BY-NC-SA and CC0),
  and the packaged filename carries the OFO mission id, so one override row per mission
  assigns that plot's license. A source-level license covers everything the overrides do
  not, and it is deliberately the *most restrictive* of the licenses a mixed source spans.

Filtering removes annotation rows. It never re-splits anything: an image whose annotations
are dropped simply leaves the split it was already in, and every remaining image keeps its
original train/validation/test assignment, so a license-restricted run is still scored on a
subset of the same benchmark splits.

``unknown`` is not selected by any filter except the explicit ``"unknown"`` token, so a
source with unconfirmed terms is dropped rather than silently redistributed.

CLI::

    python -m milliontrees.common.licenses                 # the license menu
    python -m milliontrees.common.licenses --csv <split.csv>   # audit a packaged split
"""

from __future__ import annotations

import fnmatch
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DATA_DIR = Path(__file__).resolve().parent / "license_data"
SOURCES_PATH = DATA_DIR / "sources.csv"
OVERRIDES_PATH = DATA_DIR / "overrides.csv"

#: License id used when a source's upstream terms have not been confirmed.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class LicenseInfo:
    """What one license permits.

    The flags are the questions a user actually filters on, not a legal summary: read the license
    itself before relying on it. ``commercial_use`` is False for the NonCommercial clause,
    ``derivatives`` False for NoDerivatives, ``share_alike`` True when a derivative must carry the
    same license, and ``copyleft`` True when the license reaches the code or model built from the
    data rather than only the data.
    """

    id: str
    name: str
    url: str
    commercial_use: bool
    derivatives: bool
    share_alike: bool
    attribution: bool
    copyleft: bool = False


LICENSES: dict[str, LicenseInfo] = {
    lic.id: lic for lic in [
        LicenseInfo(
            "CC0-1.0",
            "Creative Commons Zero v1.0 Universal",
            "https://creativecommons.org/publicdomain/zero/1.0/",
            commercial_use=True,
            derivatives=True,
            share_alike=False,
            attribution=False,
        ),
        LicenseInfo(
            "CC-BY-3.0",
            "Creative Commons Attribution 3.0",
            "https://creativecommons.org/licenses/by/3.0/",
            commercial_use=True,
            derivatives=True,
            share_alike=False,
            attribution=True,
        ),
        LicenseInfo(
            "CC-BY-4.0",
            "Creative Commons Attribution 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
            commercial_use=True,
            derivatives=True,
            share_alike=False,
            attribution=True,
        ),
        LicenseInfo(
            "CC-BY-NC-4.0",
            "Creative Commons Attribution-NonCommercial 4.0",
            "https://creativecommons.org/licenses/by-nc/4.0/",
            commercial_use=False,
            derivatives=True,
            share_alike=False,
            attribution=True,
        ),
        LicenseInfo(
            "CC-BY-NC-SA-4.0",
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            commercial_use=False,
            derivatives=True,
            share_alike=True,
            attribution=True,
        ),
        LicenseInfo(
            "CC-BY-NC-ND-3.0",
            "Creative Commons Attribution-NonCommercial-NoDerivatives 3.0",
            "https://creativecommons.org/licenses/by-nc-nd/3.0/",
            commercial_use=False,
            derivatives=False,
            share_alike=False,
            attribution=True,
        ),
        LicenseInfo(
            "CDLA-Permissive-1.0",
            "Community Data License Agreement - Permissive 1.0",
            "https://cdla.dev/permissive-1-0/",
            commercial_use=True,
            derivatives=True,
            share_alike=False,
            attribution=True,
        ),
        LicenseInfo(
            "AGPL-3.0-or-later",
            "GNU Affero General Public License v3.0 or later",
            "https://www.gnu.org/licenses/agpl-3.0",
            commercial_use=True,
            derivatives=True,
            share_alike=True,
            attribution=True,
            copyleft=True,
        ),
        LicenseInfo(
            UNKNOWN,
            "Upstream terms not confirmed",
            "",
            commercial_use=False,
            derivatives=False,
            share_alike=False,
            attribution=True,
        ),
    ]
}

# Spelling variants seen in the manuscript table, upstream READMEs and the packaged
# metadata, keyed by their squashed form (lowercase, punctuation removed).
_ALIASES = {
    "cc0": "CC0-1.0",
    "cc010": "CC0-1.0",
    "cc01": "CC0-1.0",
    "publicdomain": "CC0-1.0",
    "ccby3": "CC-BY-3.0",
    "ccby30": "CC-BY-3.0",
    "ccby4": "CC-BY-4.0",
    "ccby40": "CC-BY-4.0",
    "ccbync4": "CC-BY-NC-4.0",
    "ccbync40": "CC-BY-NC-4.0",
    "ccbyncsa": "CC-BY-NC-SA-4.0",
    "ccbyncsa4": "CC-BY-NC-SA-4.0",
    "ccbyncsa40": "CC-BY-NC-SA-4.0",
    "ccbyncnd": "CC-BY-NC-ND-3.0",
    "ccbyncnd3": "CC-BY-NC-ND-3.0",
    "ccbyncnd30": "CC-BY-NC-ND-3.0",
    "cdlapermissive": "CDLA-Permissive-1.0",
    "cdlapermissive1": "CDLA-Permissive-1.0",
    "cdlapermissive10": "CDLA-Permissive-1.0",
    "agpl3": "AGPL-3.0-or-later",
    "agpl30": "AGPL-3.0-or-later",
    "agpl30orlater": "AGPL-3.0-or-later",
    "agpl30gpl30": "AGPL-3.0-or-later",
    "agpl30gpl30orlater": "AGPL-3.0-or-later",
    "unknown": UNKNOWN,
    "": UNKNOWN,
    "nan": UNKNOWN,
    "none": UNKNOWN,
}


def _squash(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_license(value) -> str:
    """Canonical license id for a free-text license string.

    ``"CC BY 4.0"``, ``"CC-BY 4.0"`` and ``"cc_by_4"`` all resolve to ``"CC-BY-4.0"``.
    """
    squashed = _squash(value)
    if squashed in _ALIASES:
        return _ALIASES[squashed]
    for lic_id in LICENSES:
        if _squash(lic_id) == squashed:
            return lic_id
    raise ValueError(
        f"Unrecognized license {value!r}. Known licenses: {sorted(LICENSES)}. "
        "Add it to milliontrees.common.licenses.LICENSES if it is a real upstream license."
    )


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def _ids(predicate) -> frozenset[str]:
    return frozenset(lic.id
                     for lic in LICENSES.values()
                     if lic.id != UNKNOWN and predicate(lic))


#: Named groups a user can ask for instead of listing license ids. Every preset except
#: ``all`` and ``unknown`` excludes ``unknown``, so unconfirmed sources are never included
#: by accident.
PRESETS: dict[str, frozenset[str]] = {
    "all":
        frozenset(LICENSES),
    "known":
        _ids(lambda lic: True),
    "commercial":
        _ids(lambda lic: lic.commercial_use),
    "noncommercial":
        _ids(lambda lic: not lic.commercial_use),
    "derivatives":
        _ids(lambda lic: lic.derivatives),
    "no-share-alike":
        _ids(lambda lic: not lic.share_alike),
    "no-copyleft":
        _ids(lambda lic: not lic.copyleft),
    "permissive":
        _ids(lambda lic: lic.commercial_use and lic.derivatives and not lic.
             share_alike and not lic.copyleft),
    "public-domain":
        _ids(lambda lic: not lic.attribution),
    "unknown":
        frozenset({UNKNOWN}),
}


def resolve_selection(selection) -> frozenset[str]:
    """License ids named by ``selection``.

    ``selection`` is a preset name (``PRESETS``), a license id, an fnmatch pattern over license ids
    (``"CC-BY-*"``), or any sequence of those; a sequence selects their union. ``None`` means no
    filtering and returns every id, ``unknown`` included.
    """
    if selection is None:
        return PRESETS["all"]
    tokens: Sequence = ([selection]
                        if isinstance(selection, str) else list(selection))
    if not tokens:
        return PRESETS["all"]

    allowed: set[str] = set()
    for token in tokens:
        key = str(token).strip()
        if key.lower() in PRESETS:
            allowed |= PRESETS[key.lower()]
            continue
        if any(ch in key for ch in "*?["):
            matched = {
                lic_id for lic_id in LICENSES
                if fnmatch.fnmatch(lic_id.lower(), key.lower())
            }
            if not matched:
                raise ValueError(
                    f"License pattern {token!r} matched nothing. Known licenses: "
                    f"{sorted(LICENSES)}")
            allowed |= matched
            continue
        allowed.add(normalize_license(key))
    return frozenset(allowed)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _source_key(value) -> str:
    return " ".join(str(value).split()).lower()


def load_source_licenses(path=None) -> dict[str, str]:
    """``{normalized source name: license id}`` from ``sources.csv``."""
    import pandas as pd

    table = pd.read_csv(path or SOURCES_PATH, comment="#")
    return {
        _source_key(row.source): normalize_license(row.license)
        for row in table.itertuples()
    }


def load_license_overrides(path=None):
    """Per-row override rules from ``overrides.csv``, in file order."""
    import pandas as pd

    table = pd.read_csv(path or OVERRIDES_PATH, comment="#")
    return [(_source_key(row.source), str(row.filename_pattern),
             normalize_license(row.license)) for row in table.itertuples()]


def row_licenses(df, source_licenses=None, overrides=None, warn=True):
    """License id for every row of ``df``, as a ``pandas.Series`` aligned to its index.

    ``df`` needs a ``source`` column; ``filename`` is used when overrides apply. Sources missing
    from the table resolve to ``unknown`` and raise a warning, so a source added to a release
    without a license entry is visible instead of being quietly excluded.
    """
    import pandas as pd

    source_licenses = (load_source_licenses()
                       if source_licenses is None else source_licenses)
    overrides = load_license_overrides() if overrides is None else overrides

    # Resolve per *unique* source and then map the column once. The box release is
    # 7.5M rows against ~20 sources, so normalizing every row's string individually
    # costs seconds for nothing.
    source_column = df["source"].astype(str)
    unique_sources = source_column.unique()
    keys = {value: _source_key(value) for value in unique_sources}
    resolved = {value: source_licenses.get(key) for value, key in keys.items()}
    result = source_column.map(resolved)

    missing = sorted(
        key for value, key in keys.items() if resolved[value] is None)
    if missing and warn:
        warnings.warn(
            f"No license recorded for source(s) {missing}; treating them as "
            f"'{UNKNOWN}', which every license filter excludes. Add them to "
            f"{SOURCES_PATH}.",
            stacklevel=2,
        )
    result = result.fillna(UNKNOWN)

    if overrides and "filename" in df.columns:
        filenames = df["filename"].astype(str)
        for source_key, pattern, license_id in overrides:
            in_source = source_column.map({
                value: key == source_key for value, key in keys.items()
            })
            if not in_source.any():
                continue
            match = in_source & filenames.map(
                lambda f, p=pattern: fnmatch.fnmatch(f, p))
            if match.any():
                result = result.mask(match, license_id)

    return pd.Series(result, index=df.index, name="license")


def source_license_map(df, source_licenses=None, overrides=None):
    """``{source: [license ids]}`` for the sources present in ``df``.

    Resolved from the unique source names rather than row by row, so it is cheap on a multi-million-
    row release. A source that ``overrides.csv`` splits lists every license its rows can carry; use
    :func:`row_licenses` when the per-row answer is what is needed.
    """
    source_licenses = (load_source_licenses()
                       if source_licenses is None else source_licenses)
    overrides = load_license_overrides() if overrides is None else overrides

    present = {
        value: _source_key(value) for value in df["source"].astype(str).unique()
    }
    result = {
        value: {source_licenses.get(key, UNKNOWN)}
        for value, key in present.items()
    }
    for source_key, _pattern, license_id in overrides:
        for value, key in present.items():
            if key == source_key:
                result[value].add(license_id)
    return {value: sorted(ids) for value, ids in result.items()}


def filter_by_license(df, selection, verbose=False):
    """Drop annotation rows whose license is not in ``selection``.

    Returns ``(filtered_df, licenses)`` where ``licenses`` is the per-row license of the *input*
    frame. ``selection=None`` returns ``df`` unchanged. Splits are untouched: rows are removed,
    nothing is reassigned.
    """
    licenses = row_licenses(df)
    if selection is None:
        return df, licenses

    allowed = resolve_selection(selection)
    keep = licenses.isin(allowed)

    if verbose:
        dropped = licenses[~keep]
        print(f"[MillionTrees] licenses={selection} -> {sorted(allowed)}")
        print(
            f"[MillionTrees] kept {int(keep.sum())}/{len(df)} annotations "
            f"from {df.loc[keep, 'source'].nunique()}/{df['source'].nunique()} sources"
        )
        if not dropped.empty:
            for license_id, count in dropped.value_counts().items():
                sources = sorted(
                    set(df.loc[dropped.index, "source"][dropped == license_id]))
                print(f"[MillionTrees]   dropped {count} annotations under "
                      f"{license_id}: {', '.join(sources)}")

    if not keep.any():
        raise ValueError(
            f"licenses={selection} matched no annotations. Resolved to "
            f"{sorted(allowed)}; the data carries {sorted(licenses.unique())}.")

    return df[keep], licenses


def license_summary(df):
    """Per-license annotation, image and source counts for ``df``."""
    import pandas as pd

    licenses = row_licenses(df, warn=False)
    frame = pd.DataFrame({
        "license":
            licenses,
        "source":
            df["source"].astype(str),
        "filename":
            df["filename"].astype(str) if "filename" in df.columns else "",
    })
    summary = frame.groupby("license").agg(
        annotations=("source", "size"),
        images=("filename", "nunique"),
        sources=("source", "nunique"),
    )
    return summary.sort_values("annotations", ascending=False)


def describe_presets() -> str:
    """Human-readable menu of presets and the licenses each selects."""
    lines = ["Presets (pass as licenses=...):"]
    for name, ids in PRESETS.items():
        lines.append(f"  {name:<14} {', '.join(sorted(ids))}")
    lines.append("")
    lines.append("Licenses:")
    for lic in LICENSES.values():
        flags = ", ".join(flag for flag, on in [
            ("commercial", lic.commercial_use),
            ("derivatives", lic.derivatives),
            ("share-alike", lic.share_alike),
            ("attribution", lic.attribution),
            ("copyleft", lic.copyleft),
        ] if on) or "no permissions assumed"
        lines.append(f"  {lic.id:<20} {flags}")
    return "\n".join(lines)


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        help=
        "Packaged split CSV to audit (e.g. TreeBoxes_v0.24/within-distribution.csv)"
    )
    parser.add_argument("--licenses",
                        nargs="*",
                        help="Selection to report on, e.g. commercial")
    args = parser.parse_args(argv)

    if args.csv is None:
        print(describe_presets())
        return 0

    import pandas as pd

    df = pd.read_csv(args.csv, low_memory=False)
    print(license_summary(df).to_string())
    if args.licenses:
        print()
        filter_by_license(df, args.licenses, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
