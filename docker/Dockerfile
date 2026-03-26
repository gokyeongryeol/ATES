FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /workspace/ATES

# Install the required packages
RUN apt-get update \
    && apt-get install -y vim ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 build-essential gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel ninja

RUN pip install "huggingface_hub[cli]"

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/install_editable.sh ./scripts/install_editable.sh

RUN bash scripts/install_editable.sh base
