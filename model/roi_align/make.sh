#!/usr/bin/env bash

#CUDA_PATH=/usr/local/cuda/
CUDA_PATH=/home/khosungpil/anaconda3/envs/torch04/lib/python3.7/site-packages/torch/cuda
CUDA_ARCH="-gencode arch=compute_30, code=sm_30 \
           -gencode arch=compute_35, code=sm_35 \
           -gencode arch=compute_50, code=sm_50 \
           -gencode arch=compute_52, code=sm_52 \
           -gencode arch=compute_60, code=sm_60 \
           -gencode arch=compute_61, code=sm_61 "

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py
