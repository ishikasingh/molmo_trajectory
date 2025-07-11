FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

# System dependencies
RUN apt update && \
    apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt install -y netcat dnsutils && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx git libvulkan-dev \
    zip unzip wget curl git git-lfs build-essential cmake \
    vim less sudo htop ca-certificates man tmux ffmpeg tensorrt \
    # Add OpenCV system dependencies
    libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip setuptools
RUN pip install gpustat wandb==0.19.0
# Create and set working directory
WORKDIR /workspace
# Copy pyproject.toml for dependencies
COPY pyproject.toml .
# Install dependencies from pyproject.toml
RUN pip install -e .[base]
# There's a conflict in the native python, so we have to resolve it by
RUN pip uninstall -y transformer-engine
RUN pip install flash_attn==2.7.1.post4 -U --force-reinstall
# Clean any existing OpenCV installations
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 || true
RUN pip install opencv-python==4.8.0.74
RUN pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 numpy==1.26.4
COPY getting_started /workspace/getting_started
COPY scripts /workspace/scripts
COPY demo_data /workspace/demo_data
RUN pip install -e . --no-deps
# need to install accelerate explicitly to avoid version conflicts
RUN pip install accelerate>=0.26.0

# Install AWS ClI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip ./aws

# Enable github ssh access
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# # Set environment variables 
RUN echo "export PYTHONPATH=$PYTHONPATH/" >> "/root/.bashrc" && \
    echo "export HF_HOME=/root/.cache/huggingface" >> "/root/.bashrc" && \
    echo "export HF_HUB_CACHE=/root/.cache/huggingface/hub" >> "/root/.bashrc"

RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh && \
    bash ./Miniforge3-25.3.0-3-Linux-x86_64.sh -b -p $HOME/miniforge3 -s && \
    rm Miniforge3-25.3.0-3-Linux-x86_64.sh && \
    eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda init && conda config --set auto_activate_base true && conda activate base && \
    mamba shell init --shell bash  && \
    conda create --name handobj_new python=3.8 -y && \
    conda activate handobj_new && \
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch && \
    conda deactivate && \
    conda create --name sam python=3.12 -y && \
    pip install --upgrade setuptools && \
    pip install -e .[base] && \
    pip install --no-build-isolation flash-attn==2.7.1.post4 

    # conda create -n dexmimicgen python=3.10 -y && \
    # conda activate dexmimicgen && \
    # conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
    # cd robosuite && \
    # pip install -e . && \
    # cd ../FAR-dexmimicgen && \
    # pip install -e . && \
    # pip install numpy==1.24 && \
    # pip install ipdb

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]