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

Standard AWS credential variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
are used by the AWS SDK.

### Responses

| Status | When                                            |
| ------ | ----------------------------------------------- |
| 200    | Success.                                        |
| 400    | Upload is not a recognised image.               |
| 413    | Upload exceeds `MAX_UPLOAD_BYTES`.              |
| 415    | HEIC/HEIF image — export as JPEG and retry.      |
| 502    | Object storage upload failed.                   |

## Development

```bash
pip install -r requirements-dev.txt
pytest          # or: npm run test
mypy app        # or: npm run typecheck
```
