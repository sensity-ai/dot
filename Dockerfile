FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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

# Create and activate the conda environment, install requirements and download and extract the checkpoints
RUN conda init bash \
    && . ~/.bashrc \
    && cd dot && conda env create -f envs/environment-gpu.yaml \
    && conda activate dot \
    && pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -e . \
    && pip install gdown \
    && gdown 1Qaf9hE62XSvgmxR43dfiwEPWWS_dXSCE \
    && unzip -o dot_model_checkpoints.zip \
    && rm -rf *.z*

RUN cd dot \
    && /root/miniconda3/envs/dot/bin/pip install --no-cache-dir -e .

ENTRYPOINT /bin/bash
