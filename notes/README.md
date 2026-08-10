# Working notes

Internal analysis write-ups, diagnostic reports, experiment logs and generated table
fragments. These are working documents for maintainers and agents — they record *why* a
number is what it is, which job produced it, and what was tried and rejected.

**They are deliberately not part of the published documentation.** `docs/` is the
readthedocs site and only contains pages a benchmark user would read; every page there is
in the `docs/index.rst` toctree, and the docs build runs with `-W`, so an unreferenced
markdown file in `docs/` fails CI. Put analysis and scratch write-ups here instead.

Conventions:

- One file per investigation or generated table. Scripts under `scripts/` that emit a
  report write it here (e.g. `make_ap_iou_table.py`, `aggregate_threshold_sweep.py`,
  `make_weak_supervision_report.py`).
- Figures still live in `docs/public/` so the published pages can use them; notes link to
  them as `../docs/public/...`.
- If a note graduates into something a benchmark user should read, move it into `docs/`
  *and* add it to the toctree in `docs/index.rst`.
- `notes/` is excluded from the sdist (see `MANIFEST.in`).
