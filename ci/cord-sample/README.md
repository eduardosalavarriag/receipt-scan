# Fixed CORD sample — accuracy guard (#10)

A small, fixed sample of receipts in the CORD (NAVER Clova) dataset format, used
by the CI **accuracy guard**. On every pull request, CI runs the extractor over
this sample and fails the build if **total** accuracy drops below the baseline in
`baseline.json` — so a change that makes reading worse is caught before merge.

## Layout

```
labels/<id>.json   CORD-format labels: {"source": "cord", "company", "total", "words": [{text, box}]}
baseline.json      {"total_accuracy": ...} — the score the guard protects
```

The guard reads the stored OCR `words` only (see `app/evaluate.py`); it needs no
images and no network, so it is fully deterministic in CI.

## Why these labels are committed stand-ins

CORD is licensed by NAVER Clova (CC BY 4.0). Rather than redistribute slices of
the dataset in this repo, the labels here are small CORD-*format* receipts that
stand in for a real CORD slice. To guard against a genuine CORD slice instead,
`download` the dataset (see the README) and point the guard at that data
directory — the format is identical.

## Updating the baseline

`baseline.json` is committed, so changing `total_accuracy` is a deliberate,
reviewable diff. Raise it only when an improvement is intended and real.
