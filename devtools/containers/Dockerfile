# Use the Ubuntu base image and copy in the uv binary
FROM ubuntu:24.04

LABEL org.opencontainers.image.source=https://github.com/OpenADMET/openadmet-models
LABEL org.opencontainers.image.description="OpenADMET models"
LABEL org.opencontainers.image.licenses=MIT

# Torch build for the image: cuda (CUDA 12.6 wheels) or cpu
ARG TORCH_EXTRA=cuda

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/home/ubuntu/python \
    UV_PROJECT_ENVIRONMENT=/home/ubuntu/venv \
    PATH=/home/ubuntu/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /usr/local/bin/

# git fetches the git dependencies pinned in uv.lock; libgomp1 is the OpenMP runtime that the LightGBM wheel links
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create the project directory as the ubuntu user so editable installs can write build metadata there
USER ubuntu
RUN mkdir -p /home/ubuntu/openadmet-models
WORKDIR /home/ubuntu/openadmet-models

# Install locked dependencies first so source edits do not invalidate this layer
COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --extra all --extra ${TORCH_EXTRA}

# Install the project in editable mode
COPY --chown=ubuntu:ubuntu . .
RUN uv sync --locked --extra all --extra ${TORCH_EXTRA}

WORKDIR /home/ubuntu
