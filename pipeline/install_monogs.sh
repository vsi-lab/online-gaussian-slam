#!/usr/bin/env bash
# Build a MonoGS env for the RTX 5080 (Blackwell, sm_120).
# Upstream pins CUDA 11.6 / PyTorch 1.12 / Python 3.7, none of which support sm_120.
# Changes vs upstream environment.yml are logged in notes/monogs_input_spec.md.
set -eo pipefail   # no -u: conda activate.d scripts reference unset vars
source ~/miniconda3/etc/profile.d/conda.sh
ENV=monogs

if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n $ENV --override-channels -c conda-forge python=3.10
fi
conda activate $ENV

# CUDA 12.8 compiler inside the env (system nvcc is 12.0, too old for sm_120)
conda install -y --override-channels -c "nvidia/label/cuda-12.8.1" -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -1

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

pip install "numpy<2" plyfile tqdm opencv-python==4.10.0.84 munch trimesh evo==1.11.0 matplotlib==3.5.3 open3d==0.18.0 \
  torchmetrics imgviz PyOpenGL glfw PyGLM wandb lpips rich ruff

cd ~/MonoGS
# simple-knn uses FLT_MAX without <cfloat>; newer compilers reject it
grep -q '#include <cfloat>' submodules/simple-knn/simple_knn.cu || \
  sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu
# rasterizer uses uint32_t/uint64_t without <cstdint>; newer compilers reject it
grep -q '#include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h || \
  sed -i '0,/#pragma once/s//#pragma once\n#include <cstdint>/' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation submodules/simple-knn submodules/diff-gaussian-rasterization

python -c "import torch, simple_knn, diff_gaussian_rasterization; print('MonoGS env OK')"
