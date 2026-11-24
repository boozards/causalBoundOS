FROM python:3.13.2-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DIR=/app

# Exact minor-version dependency lock matching serialization environment
RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 \
    numpy==2.2.3 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    joblib==1.4.2 \
    pydantic==2.10.6 \
    pytest==8.3.4

# Create non-root execution identity (UID 1000)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy read-only benchmark input assets owned by root (unwritable by appuser)
COPY environment/data/ /app/data/
RUN chmod -R a-w,a+rX /app/data && \
    mkdir -p /app/output && \
    chown -R appuser:appuser /app/output && \
    chown appuser:appuser /app

USER appuser

CMD ["/bin/bash"]
