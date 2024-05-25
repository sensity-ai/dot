FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# copy repo codebase
COPY . ./dot

# set working directory
WORKDIR ./dot

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

# Install requirements
RUN conda config --add channels conda-forge
RUN conda install python==3.8
RUN conda install pip==21.3
RUN pip install onnxruntime-gpu==1.9.0
RUN pip install -r requirements.txt

# Install pytorch
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install dot
RUN pip install -e .

# Download and extract the checkpoints
RUN pip install gdown
RUN gdown 1Qaf9hE62XSvgmxR43dfiwEPWWS_dXSCE
RUN unzip -o dot_model_checkpoints.zip
RUN rm -rf *.z*

ENTRYPOINT /bin/bash
