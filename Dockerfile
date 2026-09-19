FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.5 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main \
    && rm -rf ~/.cache/pypoetry ~/.cache/pip

COPY .project-root LICENSE README.md CITATION.cff ./
COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests

RUN git config --global --add safe.directory /app
