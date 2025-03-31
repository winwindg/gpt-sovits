#!/bin/bash

# set current directory
project_name="gpt_sovits"

# check environment
if ! conda info --envs | grep -q $project_name; then
    echo "Creating conda environment: $project_name"
    conda create -n $project_name python=3.9 -y
else
    echo "Conda environment $project_name already exists"
fi

# activate environment
echo "Activating conda environment: $project_name"
source activate $project_name

# install dependencies
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip install --upgrade pip
pip install /mnt/qnap/aigc/wheel/torch-2.1.1+cu118-cp39-cp39-linux_x86_64.whl
pip install torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-extra.txt
pip install -r requirements.txt

mkdir -p input
mkdir -p output/asr_opt
mkdir -p output/denoise_opt
mkdir -p output/slicer_opt
mkdir -p output/uvr5_opt

cp -rf /mnt/qnap/aigc/projects/gpt-sovits-v2/* .
