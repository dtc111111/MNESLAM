FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# For RTX 30 series, use 86
ENV TCNN_CUDA_ARCHITECTURES=86
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN sed -i 's|http://.*.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    wget git curl bzip2 ca-certificates sudo ninja-build build-essential \
    libgl1-mesa-glx libglib2.0-0 libopenexr-dev cmake && \
    apt-get clean

# Install Python、pip
RUN apt update && apt install -y \
    python3 python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p ~/.pip && echo "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/" > ~/.pip/pip.conf

RUN pip install --upgrade pip

# Install PyTorch（cu113）
RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install pytorch_scatter
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

# Install PyTorch3D
RUN pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3.8_cu113_pyt1100/download.html

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -r requirements.txt

# Install thirdparty packages
# RUN pip install thirdparty/lietorch
# RUN pip install .
#
# # Build PyTorch3D
# RUN pip install git+https://github.com/facebookresearch/pytorch3d.git
#
# # Build tinycudann
# RUN pip install git+https://github.com/nvlabs/tiny-cuda-nn

# # If you cannot access network when you use GPUs
# RUN git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
# COPY tiny-cuda-nn /workspace/tiny-cuda-nn

# # # Try this version if you cannot use the latest version of tinycudann
# # WORKDIR /workspace/tiny-cuda-nn/
# # RUN git reset --hard 91ee479d275d322a65726435040fc20b56b9c991

# WORKDIR /workspace/tiny-cuda-nn/bindings/torch
# RUN pip install .

RUN rm -rf /workspace

CMD ["bash"]
