# Use the official NVIDIA CUDA base image for Ubuntu 22.04 with CUDA 12.2
# FROM nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

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

# Set CUDA environment variables (base image should have CUDA in /usr/local/cuda)
# ENV CUDA_HOME=/usr/local/cuda-12.1
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# # Verify CUDA installation and add to bash profile
# RUN echo "export CUDA_HOME=/usr/local/cuda-12.1" >> /root/.bashrc && \
#     echo "export PATH=/usr/local/cuda-12.1/bin:\$PATH" >> /root/.bashrc && \
#     echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc

#     WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Verify CUDA installation and add to bash profile
RUN echo "export CUDA_HOME=/usr/local/cuda-12.8" >> /root/.bashrc && \
    echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc

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

# Download and install miniconda instead of miniforge
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 -s && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda and configure channels
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda init && \
    conda config --set auto_activate_base true && \
    conda config --set channel_priority flexible && \
    echo "yes" | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    echo "yes" | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and setup handobj_new environment
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda create --name handobj_new python=3.8 -y

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate handobj_new && \
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate handobj_new && \
    conda install -c conda-forge cudatoolkit=11.3 cudatoolkit-dev=11.3

# RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
#     conda activate handobj_new && \
#     conda install gcc_linux-64=9 gxx_linux-64=9

# Create and setup sam environment
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda create --name sam python=3.11 -y

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    conda install pytorch>=2.5.1 torchvision>=0.20.1 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install sam2 package
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[notebooks]"

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[interactive-demo]"

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate sam && \
    cd vendor/sam2 && \
    pip install -e ".[dev]"

# Create and setup molmo environment
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda create --name molmo python=3.10 -y

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install molmo package
RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    cd molmo && \
    pip install -e .

RUN eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
    conda activate molmo && \
    cd molmo && \
    pip install -e ".[train,serve]" && \
    pip uninstall -y torch torchvision && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]