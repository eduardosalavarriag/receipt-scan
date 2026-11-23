# receipt-scan

Scan a phone photo of a receipt and get back a cleaned-up, grayscale image.

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

It takes a JPEG, PNG, or WebP photo and returns a **cropped, straightened,
lighting-corrected grayscale** image of the receipt. The image is *not*
binarised to pure black and white — the full tonal range is kept so later OCR
stages have more to work with.

By default the image comes back inline as base64 in the JSON body:

```json
{ "image": "iVBORw0KGgo..." }
```

Pass `?image=false` to omit it.

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
