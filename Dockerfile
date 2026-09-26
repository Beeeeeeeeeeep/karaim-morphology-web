FROM python:3.12-slim
WORKDIR /app
COPY app_bundle.tar.gz.b64 /tmp/app_bundle.tar.gz.b64
COPY v113_patch.tar.gz.b64 /tmp/v113_patch.tar.gz.b64
COPY apply_unified_patch.py /tmp/apply_unified_patch.py
RUN base64 -d /tmp/app_bundle.tar.gz.b64 > /tmp/app_bundle.tar.gz \
 && tar -xzf /tmp/app_bundle.tar.gz -C /app \
 && base64 -d /tmp/v113_patch.tar.gz.b64 > /tmp/v113_patch.tar.gz \
 && tar -xzf /tmp/v113_patch.tar.gz -C /tmp \
 && python /tmp/apply_unified_patch.py /app/karaim_morph_engine_v1_1.py /tmp/v113_engine.patch \
 && python /tmp/apply_unified_patch.py /app/app.py /tmp/v113_app.patch \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && rm -f /tmp/app_bundle.tar.gz /tmp/app_bundle.tar.gz.b64 /tmp/v113_patch.tar.gz /tmp/v113_patch.tar.gz.b64 /tmp/v113_engine.patch /tmp/v113_app.patch /tmp/apply_unified_patch.py
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
