# Use the official CUDA 12.3 base image from NVIDIA
FROM nvidia/cuda:12.3.0-base-ubuntu20.04

 # Set timezone to Asia/Taipei
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio

# Install Jupyter Notebook
RUN pip3 install jupyter 

RUN pip3 install matplotlib scipy tqdm 
# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set up workspace directory
RUN mkdir /workspace
WORKDIR /workspace

# Copy your project files into the container (assuming Docker context includes them)
# COPY . /workspace


# Expose the port Jupyter Notebook will run on
EXPOSE 8888

# Command to start Jupyter Notebook with no browser, accessible from any IP
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]