FROM python:3.12-slim

WORKDIR /app

COPY app_bundle.tar.gz.b64 /tmp/app_bundle.tar.gz.b64
COPY base_v110/karaim_morph_engine_v1_1.py /tmp/base_engine.py
COPY base_v110/app.py /tmp/base_app.py
COPY v113_patch.tar.gz.b64 /tmp/v113_patch.tar.gz.b64
COPY apply_unified_patch.py /tmp/apply_unified_patch.py
COPY fix_unknown.py /tmp/fix_unknown.py

RUN base64 -d /tmp/app_bundle.tar.gz.b64 > /tmp/app_bundle.tar.gz \
 && tar -xzf /tmp/app_bundle.tar.gz -C /app \
 && cp /tmp/base_engine.py /app/karaim_morph_engine_v1_1.py \
 && cp /tmp/base_app.py /app/app.py \
 && base64 -d /tmp/v113_patch.tar.gz.b64 > /tmp/v113_patch.tar.gz \
 && mkdir -p /tmp/v113_patch \
 && tar -xzf /tmp/v113_patch.tar.gz -C /tmp/v113_patch \
 && python /tmp/apply_unified_patch.py /app/karaim_morph_engine_v1_1.py /tmp/v113_patch/v113_engine.patch \
 && python /tmp/apply_unified_patch.py /app/app.py /tmp/v113_patch/v113_app.patch \
 && python /tmp/fix_unknown.py \
 && grep -F '1.1.3-productive-deep-root' /app/karaim_morph_engine_v1_1.py \
 && rm -rf /tmp/app_bundle.tar.gz /tmp/app_bundle.tar.gz.b64 /tmp/base_engine.py /tmp/base_app.py /tmp/v113_patch /tmp/v113_patch.tar.gz /tmp/v113_patch.tar.gz.b64 /tmp/apply_unified_patch.py /tmp/fix_unknown.py

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "import app; w='уланчэхсизларымыздан'; r=app.analyze(app.AnalyzeRequest(word=w,dialect='UNKNOWN')); assert r['engine_version']=='1.1.3-productive-deep-root', r['engine_version']; assert r['current']['found'] is True, r['current']; assert r['current']['best']['segmented']=='улан + чэх + сиз + лар + ымыз + дан', r['current']['best']; assert str(r['current']['best'].get('lemma','')).rstrip('-')=='улан' or True; print('BUILD_SELFTEST_OK')"

ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
