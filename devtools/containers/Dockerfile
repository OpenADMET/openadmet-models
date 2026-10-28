# Use the conda-forge base image with micromamba
FROM mambaorg/micromamba:2-ubuntu24.04

LABEL org.opencontainers.image.source=https://github.com/OpenADMET/openadmet-models
LABEL org.opencontainers.image.description="OpenADMET models"
LABEL org.opencontainers.image.licenses=MIT

# set environment variables
ENV PYTHONUNBUFFERED=1

# Lock file to install, relative to the repository root; the CUDA builds are pinned inside it
ARG LOCK_FILE=devtools/locks/py312-cuda-lock.yml
# Lock categories to install (the full dependency group, see docs/installation.rst)
ARG LOCK_CATEGORIES="--category main --category full"

# micromamba only recognizes lock files whose name ends in -lock.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER ${LOCK_FILE} /tmp/openadmet-models-lock.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER devtools/scripts/install_pip_layer.py /tmp/devtools/scripts/
COPY --chown=$MAMBA_USER:$MAMBA_USER devtools/conda-envs/pip-*.txt /tmp/devtools/conda-envs/

RUN micromamba install -y -n base -f /tmp/openadmet-models-lock.yml ${LOCK_CATEGORIES} && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# The pip layer contains the git dependencies; git is part of the lock file
RUN python /tmp/devtools/scripts/install_pip_layer.py --group full

COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/openadmet-models

WORKDIR /home/mambauser

RUN python -m pip install --no-deps -e openadmet-models && \
    python -m pip check
