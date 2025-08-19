#!/usr/bin/env bash

PYTHON='python'
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# export CUDA_HOME='/usr/local/cuda-11.5/'
# export PATH=/usr/local/cuda-11-5/bin:$PATH
# export LD_LIBRARY_PATH="/usr/local/cuda-11.5/lib64":$LD_LIBRARY_PATH



echo "Building roi align op..."
cd mmdet/ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd mmdet/ops/roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd mmdet/ops/nms
make clean
make PYTHON=${PYTHON}

echo "Building dconv op..."
cd mmdet/ops/dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building correlation op..."
cd mmdet/ops/correlation
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building crop_ops..."
cd mmdet/ops/crop
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace


echo "Building sigmoid_focal_loss..."
cd mmdet/ops/sigmoid_focal_loss
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

