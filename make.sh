#!/usr/bin/env bash

# cython==0.25.2
#CUDA_PATH=/usr/local/cuda/
CUDA_PATH=/home/khosungpil/anaconda3/envs/torch04/lib/python3.7/site-packages/torch/cuda
CUDA_ARCH="-gencode arch=compute_30, code=sm_30 \
           -gencode arch=compute_35, code=sm_35 \
           -gencode arch=compute_50, code=sm_50 \
           -gencode arch=compute_52, code=sm_52 \
           -gencode arch=compute_60, code=sm_60 \
           -gencode arch=compute_61, code=sm_61 "

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export PATH=$CUDA_PATH/bin:$PATH

python setup.py build_ext --inplace
rm -rf build

# compile roi_pooling
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc.."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py

# compile roi_align
cd ../../

cd model/roi_align/src
echo "Compiling roi_align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py