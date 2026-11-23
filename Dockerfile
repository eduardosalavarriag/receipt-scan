FROM python:3.11-slim

# System dependencies:
# - tesseract-ocr (+ deu/eng language data) for OCR in later stages
# - libglib2.0-0 / libgl1 are runtime deps of OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-deu \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV OCR_LANG=deu+eng \
    PORT=8000 \
    DATA_DIR=/data
EXPOSE 8000

# `serve` (default) runs the HTTP service; other commands (e.g. `download`)
# override the CMD: `docker run -v ./data:/data <image> download`.
ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
