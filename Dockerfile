FROM python:3.12-slim
WORKDIR /app
COPY app_bundle.tar.gz.b64 /tmp/app_bundle.tar.gz.b64
COPY v113_overlay.b64 /tmp/v113_overlay.b64
RUN base64 -d /tmp/app_bundle.tar.gz.b64 > /tmp/app_bundle.tar.gz \
 && tar -xzf /tmp/app_bundle.tar.gz -C /app \
 && base64 -d /tmp/v113_overlay.b64 > /tmp/v113_overlay.tar.gz \
 && tar -xzf /tmp/v113_overlay.tar.gz -C /app \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && rm -f /tmp/app_bundle.tar.gz /tmp/app_bundle.tar.gz.b64 /tmp/v113_overlay.tar.gz /tmp/v113_overlay.b64
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
