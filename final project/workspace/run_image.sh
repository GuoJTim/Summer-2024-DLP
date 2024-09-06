#!/bin/bash

# 设置变量
CODE_PATH="./src"
BRACKETFLARE_PATH="./BracketFlare_dataset"
RRW_PATH="./RRW_dataset"
OUTPUT_PATH="./runs"
IMAGE_NAME="final"
CONTAINER_WORKDIR="/app/src"
NATURE20_PATH="./nature20"
POSTCARD199_PATH="./postcard199"
REAL20_PATH="./real20"
SOLID200_PATH="./solid200"
WILD55_PATH="./wild55"
TRAIN_NATURE200_PATH="./train_nature200"
TRAIN_REAL89_PATH="./train_real89"

# 构建镜像
podman build -t $IMAGE_NAME .

podman run --gpus all \
            --ipc=host \
            -it \
            -v ${CODE_PATH}:${CONTAINER_WORKDIR} \
            -v ${NATURE20_PATH}:/app/nature20 \
            -v ${POSTCARD199_PATH}:/app/postcard199 \
            -v ${REAL20_PATH}:/app/real20 \
            -v ${SOLID200_PATH}:/app/solid200 \
            -v ${WILD55_PATH}:/app/wild55 \
            -v ${TRAIN_NATURE200_PATH}:/app/train_nature \
            -v ${TRAIN_REAL89_PATH}:/app/train_real \
            -v ${BRACKETFLARE_PATH}:/app/BracketFlare \
            -v ${RRW_PATH}:/app/RRW \
            -v ${OUTPUT_PATH}:/app/runs \
            -w ${CONTAINER_WORKDIR} \
            ${IMAGE_NAME} \
            /bin/bash
