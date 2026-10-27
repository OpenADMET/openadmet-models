# Ubuntu base image; pixi installs the locked full CUDA environment
FROM ubuntu:24.04

LABEL org.opencontainers.image.source=https://github.com/OpenADMET/openadmet-models
LABEL org.opencontainers.image.description="OpenADMET models"
LABEL org.opencontainers.image.licenses=MIT

# Environment from pixi.lock to install; any <group>-<hardware>-py<version> name works
ARG PIXI_ENV=full-cuda-py312
ARG PIXI_VERSION=v0.83.0

# Environment variables
ENV PYTHONUNBUFFERED=1
# Lets pixi resolve the CUDA environment on a build host without a GPU driver
ENV CONDA_OVERRIDE_CUDA="12"
ENV PIXI_HOME=/opt/pixi
ENV PATH=/opt/pixi/bin:$PATH

# Install pixi and git (pixi needs git for the pinned git dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} PIXI_NO_PATH_UPDATE=1 bash

# Copy the repository, which pixi installs in editable mode
USER ubuntu
WORKDIR /home/ubuntu
COPY --chown=ubuntu:ubuntu . openadmet-models

# Install the locked environment and write an activation script for the entrypoint
WORKDIR /home/ubuntu/openadmet-models
RUN pixi install --locked -e ${PIXI_ENV} && \
    pixi shell-hook --locked -e ${PIXI_ENV} -s bash > /home/ubuntu/activate.sh && \
    echo 'exec "$@"' >> /home/ubuntu/activate.sh && \
    pixi clean cache --yes

ENTRYPOINT ["/bin/bash", "/home/ubuntu/activate.sh"]
CMD ["openadmet", "--help"]
