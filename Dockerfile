# Single stage on purpose. The heavy dependency here is chromadb, which is a
# runtime dependency rather than a build artefact, so a builder stage would
# have to copy essentially the whole site-packages tree into the final image
# and would save nothing worth the extra moving parts.
FROM python:3.12-slim

# PYTHONUNBUFFERED so logs reach the platform's log tail as they happen rather
# than when the buffer fills; PYTHONDONTWRITEBYTECODE because a read-only
# container has nothing to gain from .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, as their own layer: requirements.txt changes far less
# often than the source, and chromadb is slow enough to install that rebuilding
# it on every code edit is the difference between a 5-second and a 3-minute
# rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# No secrets in the image. app/config.py reads everything from the environment,
# and .dockerignore excludes .env so a local one can never be baked in by
# accident. GROQ_API_KEY is supplied at run time, by --env-file locally and by
# the platform's dashboard in deployment.
ENV PORT=8000
EXPOSE 8000

# Non-root. Nothing here needs to write to the image: the Chroma index is
# EphemeralClient, built in memory at startup.
RUN useradd --create-home --uid 10001 trendly && chown -R trendly:trendly /app
USER trendly

# $PORT is read at run time, not baked in -- free-tier hosts assign it
# dynamically and a hardcoded port fails health checks.
#
# The sh -c wrapper is what expands ${PORT}; plain exec form would pass the
# literal string "$PORT" to uvicorn. `exec` then replaces the shell with
# uvicorn so it becomes PID 1 and receives SIGTERM directly -- without it the
# shell holds PID 1, swallows the signal, and the platform's graceful-stop
# window elapses into a SIGKILL on every deploy.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
