FROM debian:12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY atg/ atg/
COPY main.py .

RUN echo '#!/bin/bash\nset -e\n\nexec uv run python -m atg "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

RUN uv venv --python 3.12 && \
    uv sync

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

