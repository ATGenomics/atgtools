FROM debian:12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    pipx \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pipx install poetry
ENV PATH="/root/.local/bin:${PATH}"

COPY . .

RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-dev --no-interaction

ENTRYPOINT ["poetry", "run", "python", "-m", "atg"]
