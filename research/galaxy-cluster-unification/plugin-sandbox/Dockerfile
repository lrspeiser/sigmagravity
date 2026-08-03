FROM python:3.13.7-slim-bookworm

COPY plugin-sandbox/requirements.txt /opt/sigma-plugin/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /opt/sigma-plugin/requirements.txt \
    && useradd --no-create-home --uid 65532 --shell /usr/sbin/nologin sigmaplugin

COPY plugin-sandbox/runner.py /opt/sigma-plugin/runner.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /tmp
USER 65532:65532
ENTRYPOINT ["python", "-I", "-B", "/opt/sigma-plugin/runner.py"]
