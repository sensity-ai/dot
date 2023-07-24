FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# set working directory
WORKDIR /dot

# copy repo codebase
COPY . ./dot

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Needed by opencv
    libglib2.0-0 libsm6 libgl1 \
    libxext6 libxrender1 ffmpeg \
    build-essential cmake wget unzip zip \
    git libprotobuf-dev protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to the PATH environment variable
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN conda --version

# Download and extract model checkpoints
RUN cd dot && wget https://github.com/sensity-ai/dot/releases/download/1.0.0/dot_model_checkpoints.z01  \
    && wget https://github.com/sensity-ai/dot/releases/download/1.0.0/dot_model_checkpoints.z02 \
    && wget https://github.com/sensity-ai/dot/releases/download/1.0.0/dot_model_checkpoints.zip \
    && zip -s 0 dot_model_checkpoints.zip --out saved_models.zip \
    && unzip -o saved_models.zip \
    && rm -rf *.z*

# Create and activate the conda environment and install requirements
RUN conda init bash \
    && . ~/.bashrc \
    && cd dot && conda env create -f envs/environment-gpu.yaml \
    && conda activate dot \
    && pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN cd dot \
    && /root/miniconda3/envs/dot/bin/pip install --no-cache-dir -e .

ENTRYPOINT /bin/bash
