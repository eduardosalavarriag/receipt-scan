# receipt-scan

Scan a phone photo of a receipt and get back the store, the total, and a
cleaned-up grayscale image.

## Quickstart

```bash
docker build -t receipt-scan .
docker run --rm -p 8000:8000 receipt-scan
```

```bash
# Health check
curl http://localhost:8000/health

# Scan a receipt photo (returns the cleaned image inline as base64)
curl -X POST http://localhost:8000/scan -F "file=@receipt.jpg"

# Skip the image in the response
curl -X POST "http://localhost:8000/scan?image=false" -F "file=@receipt.jpg"
```

## What `/scan` does

It takes a JPEG, PNG, or WebP photo, cleans it up into a **cropped, straightened,
lighting-corrected grayscale** image (the full tonal range is kept — it is *not*
binarised), runs Tesseract over it, and reads the store and total with simple
rules:

- **store** — the top text line.
- **total** — the amount on a `SUMME` / `ZU ZAHLEN` / `GESAMT` / `TOTAL` line,
  otherwise the largest amount found. Amounts are parsed as exact decimals
  (European `1.234,50` and English `1,234.50` separators both work) and returned
  as a string with a dot, never a float. The currency comes from `€`/`EUR`/etc.
  on the receipt, or `null` if none is present.

Each field carries a `confidence`. A photo with no readable amount returns
`null` values with low confidence.

```json
{
  "store": { "value": "ALDI SÜD", "confidence": 0.5 },
  "total": { "value": "51.76", "currency": "EUR", "confidence": 0.5 },
  "image": "iVBORw0KGgo..."
}
```

By default the cleaned image comes back inline as base64. Pass `?image=false` to
omit it (the `store` and `total` fields are still returned).

The OCR languages default to `deu+eng` and can be changed with the `OCR_LANG`
environment variable (any Tesseract language spec, e.g. `OCR_LANG=eng`).

### Object storage (optional)

Set `S3_BUCKET` (and the related variables below) to have the processed image
uploaded to S3-compatible storage instead of returned inline. The response then
carries a key and URL:

```json
{ "key": "scans/ab12...png", "image_url": "https://.../scans/ab12...png" }
```

| Variable             | Purpose                                                   |
| -------------------- | --------------------------------------------------------- |
| `S3_BUCKET`          | Enables storage when set. Bucket to upload into.          |
| `S3_REGION`          | AWS region (default `us-east-1`).                         |
| `S3_ENDPOINT_URL`    | Custom endpoint for S3-compatible storage (e.g. MinIO).   |
| `S3_PUBLIC_URL_BASE` | Base URL used to build `image_url` (optional).            |
| `S3_KEY_PREFIX`      | Key prefix for uploaded objects (default `scans`).        |
| `MAX_UPLOAD_BYTES`   | Max accepted upload size (default 15 MB).                 |
| `OCR_LANG`           | Tesseract language spec (default `deu+eng`).              |
| `API_TOKEN`          | When set, every endpoint but `/health` requires `Authorization: Bearer <token>`. Unset leaves the service open. |
| `LABELING`           | Set `true` to enable the labelling page; requires `API_TOKEN` or the service refuses to start. |

When `API_TOKEN` is set, requests without a matching bearer token get `401`;
`/health` stays reachable without one. With `API_TOKEN` unset the service is open
(fine on a trusted home network).

Standard AWS credential variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
are used by the AWS SDK.

### Responses

| Status | When                                            |
| ------ | ----------------------------------------------- |
| 200    | Success.                                        |
| 400    | Upload is not a recognised image.               |
| 401    | `API_TOKEN` is set and the bearer token is missing or wrong. |
| 413    | Upload exceeds `MAX_UPLOAD_BYTES`.              |
| 415    | HEIC/HEIF image — export as JPEG and retry.      |
| 502    | Object storage upload failed.                   |

## Training data (`download`)

The image ships as a small command-line tool as well as a service. Fill a data
directory with training data:

```bash
docker run --rm -v ./data:/data receipt-scan download
```

This writes three sources into `/data` in one dataset format:

- **SROIE** (ICDAR 2019) and **CORD** (NAVER Clova) — public receipt datasets,
  fetched and converted.
- **Generated German receipts** — word-level rewrites of the English labels
  (`TOTAL` → `SUMME`/`ZU ZAHLEN`/`GESAMT`, `CASH` → `BAR`, `CHANGE` →
  `RÜCKGELD`, `TAX` → `MwSt`), amounts reformatted to `1.234,50 €`, German-style
  store names. These run offline from whatever English labels are already on
  disk, so the labelled store and total still appear in the receipt's own words.

Re-running `download` never duplicates receipts.

### Dataset format

```
data/
  images/<id>.jpg      the receipt image
  labels/<id>.json     {"source", "company", "total", "words": [{text, box}]}
```

`box` is an axis-aligned `[x0, y0, x1, y1]` rectangle in image pixels. `total`
is an exact decimal string with a dot (`"51.76"`) or `null`. `source` is one of
`sroie`, `cord`, `synthetic` (generated German) or `labelled`. A label may also
carry an optional `receipt_id`; several photos of one physical receipt share it
so `evaluate` folds them into a single majority vote. Point the data directory at
a different path with `DATA_DIR` (default `/data`).

## Training the ML pickers (`train`)

Once a data folder is populated, fit the store and total pickers:

```bash
docker run --rm -v ./data:/data receipt-scan train
```

`train` builds one training row per OCR candidate, holds out a slice of the
*receipts* (retakes of one receipt never straddle the split), fits two small
`MLPClassifier` networks — one for the store, one for the total — and writes
`model.joblib` into the data folder. It prints the held-out score through the same
report as `evaluate`.

Each candidate is scored on its **text** (hashed with a `HashingVectorizer`, so
the model stores **no vocabulary** on disk) plus **position and shape** features;
for totals, whether the line reads like a grand total and how its amount compares
to the largest on the receipt. The picker's probability for the winning candidate
becomes the answer's confidence.

When `model.joblib` is present, `/scan` and `evaluate` use the pickers instead of
the simple rules; otherwise they fall back to the rules automatically.

## Scoring the extractor (`evaluate`)

Score the current extractor against any data folder:

```bash
docker run --rm -v ./data:/data receipt-scan evaluate
```

It reports, for both the store and the total:

- **accuracy** — store allows small OCR typos, total is exact to the cent;
- a **per-source breakdown** (`sroie` / `cord` / `synthetic` / `labelled`);
- **retakes folded by majority vote** — photos sharing a `receipt_id` count as
  one receipt;
- an **OCR-miss vs picker-miss** split of the failures: whether the right answer
  was never among the OCR words (OCR miss) or was there but the extractor chose
  wrong (picker miss).

Run on the simple rules with no model present, it gives the baseline score.

## Accuracy guard (CI)

Every pull request runs an **accuracy guard** so a change that makes reading
worse is flagged before merge:

```bash
python -m app guard        # exits non-zero if total accuracy regresses
```

It scores the extractor over a small, fixed CORD sample committed under
`ci/cord-sample/` and fails when **total** accuracy drops below the baseline
recorded in `ci/cord-sample/baseline.json`. The guard reads the sample's stored
OCR words only — no images, no network — so it is fully deterministic in CI.

The baseline is a committed value, so raising or lowering it is a deliberate,
reviewable diff: bump `total_accuracy` only when an improvement is real and
intended. See `ci/cord-sample/README.md` for how the sample relates to CORD.

## Development

```bash
pip install -r requirements-dev.txt
pytest          # or: npm run test
mypy app        # or: npm run typecheck
```
