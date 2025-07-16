# Use the official NVIDIA CUDA base image for Ubuntu 22.04 with CUDA 12.2
# FROM nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive apt and PyTorch
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ffmpeg \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    openssh-client \
    ca-certificates \
    unzip \
    cmake \
    python3-dev python3-pip libevdev-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . ./

# Install AWS ClI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip ./aws

# Enable github ssh access
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Set environment variables 
RUN echo "export PYTHONPATH=$PYTHONPATH/" >> "/root/.bashrc" && \
    echo "export HF_HOME=/root/.cache/huggingface" >> "/root/.bashrc" && \
    echo "export HF_HUB_CACHE=/root/.cache/huggingface/hub" >> "/root/.bashrc"

# Download and install miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh && \
    bash ./Miniforge3-25.3.0-3-Linux-x86_64.sh -b -p $HOME/miniforge3 -s && \
    rm Miniforge3-25.3.0-3-Linux-x86_64.sh

# Initialize conda and mamba
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda init && \
    conda config --set auto_activate_base true && \
    conda activate base && \
    mamba shell init --shell bash

# Create and setup handobj_new environment
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda create --name handobj_new python=3.8 -y

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate handobj_new && \
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate handobj_new && \
    conda install -c conda-forge cudatoolkit=11.3 cudatoolkit-dev=11.3

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate handobj_new && \
    conda install gcc_linux-64=9 gxx_linux-64=9

# Create and setup sam environment
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda create --name sam python=3.11 -y

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    conda install pytorch>=2.5.1 torchvision>=0.20.1 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install sam2 package
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[notebooks]"

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[interactive-demo]"

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[dev]"

# Create and setup molmo environment
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda create --name molmo python=3.10 -y

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install molmo package
RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    cd molmo && \
    pip install -e .

RUN eval "$(~/miniforge3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    cd molmo && \
    pip install -e ".[train,serve]"

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]