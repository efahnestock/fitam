FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Install Python 3.10, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    sudo \
    python3-pip \
    build-essential \
    cmake \
    curl \
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
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.10 1

# Create a default user (will be modified at runtime by entrypoint)
RUN groupadd --gid 1000 developer \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash developer

RUN usermod -aG sudo developer
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir /software && chown -R developer:developer /software



USER developer

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/developer/.local/bin:$PATH"

# Install modified VTK
#WORKDIR /software
#RUN git clone -b modify-pano-rendering https://github.com/efahnestock/VTK-modified-pano.git
#WORKDIR /software/VTK-modified-pano
#RUN pip3 install .

# Create python virtualenv (so uv doesn't place one in the mounted /fitam directory)
WORKDIR /
RUN uv venv /software/env_fitam

WORKDIR /software
COPY ./learning_env_structure/ /software/learning_env_structure/
RUN sudo chown -R developer:developer /software
#WORKDIR /software/learning_env_structure
#RUN /env_fitam/bin/pip install -e .

WORKDIR /fitam
ENV UV_PROJECT_ENVIRONMENT=/software/env_fitam
COPY --chown=developer:developer pyproject.toml README.md /fitam/
COPY --chown=developer:developer src/ /fitam/src

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
RUN uv sync && rm -rf ~/.cache/uv ~/.cache/pip
ENV PATH="/software/env_fitam/bin:$PATH"
#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/software/VTK-modified-pano/build/build/lib.linux-x86_64-3.10/vtkmodules/"


# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
