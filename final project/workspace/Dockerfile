# 使用 NVIDIA PyTorch 基础映像
FROM nvcr.io/nvidia/pytorch:21.10-py3

# 设置工作目录
WORKDIR /app

# 更新 PATH 确保使用 Conda
ENV PATH=/opt/conda/bin:$PATH

# 創建並激活 Python 3.9 環境
RUN conda create -n final python=3.9 -y

RUN /bin/bash -c "source activate final"

RUN pip3 install scikit-image

# 切换到 src 文件夹
WORKDIR /app/src
