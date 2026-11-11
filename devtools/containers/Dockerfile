# Use the conda-forge base image with Python
FROM mambaorg/micromamba:2-ubuntu24.04

LABEL org.opencontainers.image.source=https://github.com/OpenADMET/openadmet-models
LABEL org.opencontainers.image.description="OpenADMET models"
LABEL org.opencontainers.image.licenses=MIT

# set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONDA_OVERRIDE_CUDA="12.4"

RUN micromamba config append channels conda-forge

COPY --chown=$MAMBA_USER:$MAMBA_USER  devtools/conda-envs/openadmet-models-gpu.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/openadmet-models

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1


WORKDIR /home/mambauser

RUN python -m pip install --no-deps -e openadmet-models
