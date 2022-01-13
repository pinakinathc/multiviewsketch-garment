#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p

source ~/.bashrc
source /vol/research/sketchcaption/miniconda/etc/profile.d/conda.sh
conda create --name garment python=3.7 numpy scikit-image -c pytorch -y

source /vol/research/sketchcaption/miniconda/etc/profile.d/conda.sh
conda activate garment

python --version
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
wget https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
pip install trimesh tqdm tensorboard future svg.path bresenham chamferdist opencv-python
conda install -c conda-forge igl -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
# conda install pytorch3d -c pytorch3d -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

sudo apt-get install libxrender1 -y
apt-get install ffmpeg libsm6 libxext6  -y

rm Miniconda3-latest-Linux-x86_64.sh
rm bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
