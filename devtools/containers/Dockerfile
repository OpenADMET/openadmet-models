# Use the conda-forge base image with Python
FROM mambaorg/micromamba:jammy

LABEL org.opencontainers.image.source=https://github.com/OpenADMET/openadmet_models
LABEL org.opencontainers.image.description="OpenADMET models"
LABEL org.opencontainers.image.licenses=MIT

# set environment variables
ENV PYTHONUNBUFFERED=1

RUN micromamba config append channels conda-forge

COPY --chown=$MAMBA_USER:$MAMBA_USER  devtools/conda-envs/openadmet_models.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/openadmet_models

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1


WORKDIR /home/mambauser

RUN python -m pip install --no-deps -e openadmet_models
