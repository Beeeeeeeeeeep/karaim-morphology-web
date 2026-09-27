FROM python:3.12-slim

WORKDIR /app

COPY app_bundle.tar.gz.b64 /tmp/app_bundle.tar.gz.b64
COPY v113_overlay.b64 /tmp/v113_overlay.b64
COPY fix_unknown.py /tmp/fix_unknown.py

RUN base64 -d /tmp/app_bundle.tar.gz.b64 > /tmp/app_bundle.tar.gz \
 && tar -xzf /tmp/app_bundle.tar.gz -C /app \
 && base64 -d /tmp/v113_overlay.b64 > /tmp/v113_overlay.tar.gz \
 && tar -xzf /tmp/v113_overlay.tar.gz -C /app \
 && python /tmp/fix_unknown.py \
 && grep -F '1.1.3-productive-deep-root' /app/karaim_morph_engine_v1_1.py \
 && rm -f /tmp/app_bundle.tar.gz /tmp/app_bundle.tar.gz.b64 /tmp/v113_overlay.tar.gz /tmp/v113_overlay.b64 /tmp/fix_unknown.py

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "import app; w='уланчэхсизларымыздан'; r=app.analyze(app.AnalyzeRequest(word=w,dialect='UNKNOWN')); assert r['engine_version']=='1.1.3-productive-deep-root', r['engine_version']; assert r['current']['found'] is True, r['current']; assert r['current']['best']['segmented']=='улан + чэх + сиз + лар + ымыз + дан', r['current']['best']; print('BUILD_SELFTEST_OK')"

ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
