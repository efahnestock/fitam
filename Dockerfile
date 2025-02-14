FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000


# Install Python 3.10, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    libxt-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libexpat1-dev \
    libxmu-dev \
    libxi-dev \
    zlib1g-dev \
    vim \
    libopenmpi-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libglib2.0-dev \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.10 1


# Install PDM
RUN pip3 install pdm

# Install modified VTK
WORKDIR /software
RUN git clone -b modify-pano-rendering https://github.com/efahnestock/VTK-modified-pano.git
WORKDIR /software/VTK-modified-pano
RUN pip3 install .


# Create python virtualenv (so pdm doesn't place one in the mounted /fitam directory)
WORKDIR /
RUN python3 -m venv /env_fitam

WORKDIR /fitam
ENV PDM_CHECK_UPDATE=false
COPY pyproject.toml pdm.lock README.md /fitam/
COPY src/ /fitam/src

RUN pdm use -f /env_fitam/
RUN pdm install
ENV PATH="/env_fitam/bin:$PATH"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/software/VTK-modified-pano/build/build/lib.linux-x86_64-3.10/vtkmodules/"


RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
