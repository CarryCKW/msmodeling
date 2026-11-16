# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# ============================================================
# msModeling NPU 完整运行镜像（基于昇腾 CANN devel 基础镜像）
#
# 基础镜像：CANN 9.0.0 devel（含 CANN toolkit / msprof / Python 3.11），
#           但【不含】torch / torch_npu / torchvision，本 Dockerfile 负责安装。
#
# 版本配套（CANN 9.0.0 官方组合，务必保持一致）：
#   CANN 9.0.0 <-> torch 2.10.0 <-> torch_npu 2.10.0.post2 <-> torchvision 0.25.0
#   同时满足 pyproject.toml 的 torch>=2.8,<=2.10 / torchvision>=0.23.0。
#
# 构建（仓库根目录）：
#   docker build -f docker/Dockerfile -t quay.io/caigentle/msmodeling:latest .
#
# 运行时必须透传 NPU 设备与宿主机驱动（Dockerfile 无法提供），示例见
# tools/perf_data_collection/README_collect_device_database.md。
#
# 关键构建期陷阱（参考 aiconfigurator-npu 的 Dockerfile.unified）：
#   1) docker build 环境没有昇腾驱动库 libascend_hal.so，构建期绝不能 import
#      torch_npu / transformers（transformers 会经 is_torch_npu_available 触发导入）。
#      安装 torch_npu 只是解包，不会导入；校验只用 importlib.metadata/find_spec。
#   2) 必须先用约束文件钉死 torch 生态版本，再 pip install -e .，
#      防止项目依赖解析把 CPU 版 torch 换成别的版本、或漏装 torch_npu。
#   3) 本镜像不含 vLLM-Ascend。采集脚本的 --version-dir 只是溯源标签，
#      不要求真实 vLLM；若 replay 用到 vLLM custom OPP，需在运行时另行挂载/安装。
# ============================================================

ARG BASE_IMAGE=quay.io/ascend/cann:9.0.0-910b-openeuler24.03-py3.11-devel
FROM ${BASE_IMAGE}

ARG PYPI_INDEX=https://pypi.org/simple
ARG PYPI_MIRROR=https://mirrors.ustc.edu.cn/pypi/web/simple
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_NPU_VERSION=2.10.0.post2

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONPATH=/workspace/msmodeling \
    HF_ENDPOINT=https://hf-mirror.com \
    TZ=Asia/Shanghai \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0

WORKDIR /workspace/msmodeling

# 1) 安装 CPU 版 torch / torchvision（torch_npu 以 CPU 版 torch 为底座，官方标准做法）。
#    用 PyTorch 官方 CPU wheel 源，避免匹配到 CUDA 版。
RUN python3 -m pip install --no-cache-dir --upgrade pip -i "${PYPI_MIRROR}" \
 && python3 -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${PYTORCH_INDEX}"

# 2) 安装 torch_npu（昇腾适配层）。主源 PyPI，附华为云昇腾镜像作为兜底。
RUN python3 -m pip install --no-cache-dir \
      "torch-npu==${TORCH_NPU_VERSION}" \
      -i "${PYPI_INDEX}" \
      --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# 3) 生成约束文件：把 torch 生态钉死在上述版本，供项目依赖解析使用。
#    只写元数据，不 import torch_npu / transformers。
RUN printf 'torch==%s\ntorchvision==%s\ntorch-npu==%s\n' \
      "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${TORCH_NPU_VERSION}" \
      > /tmp/torch-constraints.txt \
 && cat /tmp/torch-constraints.txt

# 4) 拷贝源码（.dockerignore 已排除 .git/.venv/LFS 数据等大文件）。
COPY pyproject.toml README.md ./
COPY tensor_cast/ ./tensor_cast/
COPY serving_cast/ ./serving_cast/
COPY cli/ ./cli/
COPY optix/ ./optix/
COPY web_ui/ ./web_ui/
COPY tools/ ./tools/

# 5) 可编辑安装项目：系统 Python 即运行/采集环境。约束保证 torch 生态版本不变。
RUN python3 -m pip install --no-cache-dir -i "${PYPI_MIRROR}" \
      -c /tmp/torch-constraints.txt -e . \
 && rm -f /tmp/torch-constraints.txt

# 6) 构建期只做元数据级校验：不 import transformers / torch_npu。
RUN python3 -c "import importlib.metadata as m, importlib.util as u; print('msmodeling==', m.version('msmodeling')); print('torch==', m.version('torch')); print('torch-npu==', m.version('torch-npu')); print('torch_npu spec:', u.find_spec('torch_npu') is not None)"

RUN mkdir -p /workspace/data

CMD ["/bin/bash"]
