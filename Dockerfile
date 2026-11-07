# Harness image. The jury runs it as:
#   docker run ... --read-only --tmpfs /tmp --workdir /app --entrypoint python \
#     hack:latest /app/run_agent.py --tasks-dir /data/tasks
# Everything the harness needs must be installed here: there is no network at run time
# except the LLM proxy.

# Pinned by tag and digest so rebuilds get the same base image.
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is cached while only the harness code changes.
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

COPY run_agent.py .
COPY harness/ harness/

# The image is read-only at run time, so compile bytecode now.
RUN python -m compileall -q /app
