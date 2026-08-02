FROM python:3.13.5-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY worker/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /app/requirements.txt

COPY src/voidscreen /app/src/voidscreen
COPY scripts/run_generic_field_job.py /app/run_generic_field_job.py

RUN groupadd --gid 65532 fieldworker \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin fieldworker \
    && mkdir /work \
    && chown 65532:65532 /work

USER 65532:65532
WORKDIR /work
ENTRYPOINT ["python", "/app/run_generic_field_job.py"]
