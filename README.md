# SHOW_reconstruction
Rough reconstruction of [SHOW](https://github.com/yhw-yhw/SHOW) on docker container.

This repository describes the process of reproducing [SHOW](https://github.com/yhw-yhw/SHOW) on a Docker container.  

Reproduction date: 2025/02/26  

Tested host environment:  
- Ubuntu 20.04  
- NVIDIA Driver 550.54.14  
- CUDA release 12.4, V12.4.99  
- GPU: 3× GeForce RTX 3090

Before you start, You may need to modify the environment variable (`DRIVER_V`, `DRIVER_V_MAJOR`) that specifies the driver version in the `show_prepare.sh` file.
This is used for installing `libnvidia`, so it should be matched to an available package version.


# Installation on host
```bash
git clone https://github.com/yhw-yhw/SHOW.git
cp show_prepare.sh ./SHOW/show_prepare.sh
cd SHOW

wget -q "https://www.dropbox.com/scl/fi/gwvp5c3yijkjc726bidxx/models.zip?rlkey=2p4m788qpi04oye3kur2pxszx&st=dchhjclv&dl=0" -O models.zip && unzip -qq models.zip
wget -q "https://www.dropbox.com/scl/fi/vcav90wzwqxmg56n42gr1/data.zip?rlkey=5oetna909azec027v42ogx42q&st=e5mnsldy&dl=0" -O data.zip && unzip -qq data.zip

git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Download openpose model preliminarily (optional, but recommended)
# Download method from https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
curl "https://drive.usercontent.google.com/download?id=1ZDlFGMy2Kskw3kopkbwQiUj5LG1wvE1J&confirm=t" -o openpose/models/models.zip
unzip openpose/models/models.zip -d openpose/models


git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
git checkout tags/v2.28.2
git clone https://github.com/open-mmlab/mmpose
cd mmpose
git checkout tags/v0.29.0

cd ../..
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d/
git checkout 51d7c06ddd0372728a1ef26c16fa154b605391
cd ..
```

# Run container
```bash
docker run -it --name show --gpus all --ipc=host --privileged=true -v .:/SHOW nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
```

# Prepare in container

```bash
cd /SHOW
source show_prepare.sh
```
Wait until all installations are complete.
<details>
<summary>Detail for `show_prepare.sh`</summary>

## Base setting

```bash
export TZ=Asia/Seoul
export DRIVER_V=550.54.14
export DRIVER_V_MAJOR=550


ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

apt-get update && \
    apt-get install -y \
    curl \
    git \
    unzip \
    wget \
    vim \
    libatlas-base-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    libviennacl-dev \
    libopencv-dev \
    libboost-all-dev

apt-get install -y libnvidia-common-$DRIVER_V_MAJOR=$DRIVER_V* \
    libnvidia-compute-$DRIVER_V_MAJOR=$DRIVER_V* \
    libnvidia-gl-$DRIVER_V_MAJOR=$DRIVER_V*

rm -rf /var/lib/apt/lists/*
```

## Install CMake
```bash
wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz && \
    tar -xzf cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-3.13.0-Linux-x86_64.tar.gz
```

## Install Openpose

```bash
# prevent 'dubious ownership' fatal error
git config --global --add safe.directory /SHOW/openpose
git config --global --add safe.directory /SHOW/openpose/3rdparty/caffe
git config --global --add safe.directory /SHOW/openpose/3rdparty/pybind11


cd /SHOW/openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`
```

## Install Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     bash ~/miniconda.sh -bfp /miniconda3 && \
     rm ~/miniconda.sh && \
     eval "$(/miniconda3/bin/conda shell.bash hook)"
     
conda install -y python=3.9
```

## Installation via pip and conda

```bash
cd /SHOW
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install -q --force-reinstall "cython<3.0.0"

pip install -q absl-py \
    albumentations \
    boto3 \
    cachetools \
    chumpy \
    configargparse \
    cython \
    easydict \
    einops \
    face-alignment==1.3.2 \
    facenet_pytorch==2.4.1 \
    filterpy \
    flatbuffers \
    fvcore \
    google-auth \
    google-auth-oauthlib \
    grpcio \
    h5py \
    human_body_prior \
    importlib-metadata \
    insightface \
    joblib \
    json_tricks \
    kornia \
    loguru \
    markdown \
    matplotlib \
    mediapipe \
    ninja \
    numba \
    numpy==1.23.1 \
    oauthlib \
    onnx \
    onnxruntime \
    opencv-python \
    opencv-python-headless \
    pandas \
    Pillow \
    plyfile \
    prettytable \
    protobuf \
    pyasn1 \
    pyasn1-modules \
    pycocotools \
    pynvml \
    pyrender \
    PyYAML==5.1.1 \
    qudida \
    requests \
    requests-oauthlib \
    rsa \
    scikit-image \
    scikit-learn \
    scipy \
    smplx \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorboardx \
    tensor-sensor \
    threadpoolctl \
    torchgeometry \
    tqdm \
    trimesh \
    tyro \
    wandb \
    wcwidth \
    werkzeug \
    yacs \
    yt-dlp \
    zipp

conda install -y ffmpeg=*=gpl* -c conda-forge
	
pip uninstall -y xtcocotools && pip install -q xtcocotools --no-binary xtcocotools

# prevent to downgrade torch
pip install -q --no-deps openpifpaf==0.13.8 pysparkling==0.6.2, python-json-logger

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUB_HOME=/SHOW/cub-1.10.0/cub
```

## Install MMPose

```bash
# MMPose install
cd /SHOW
pip install openmim
mim install mmcv-full==1.7.0
cd mmdetection
python setup.py install
cd mmpose
export mmpose_root=$PWD
mim install -r requirements.txt
mim install -e .

# revert numpy version
pip install -q numpy==1.23.1 

mkdir -p ~/.insightface/models
cd ~/.insightface/models
wget https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1 -O antelopev2.zip
wget https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1 -O buffalo_l.zip
mkdir -p antelopev2 && cd antelopev2 && unzip -o ../antelopev2.zip;
cd .. && mkdir -p buffalo_l && cd buffalo_l && unzip -o ../buffalo_l.zip
```

## Install Pytorch3d

```bash
cd /SHOW
cd pytorch3d/
pip install -e .
```

## Resolve path mismatch issue
```bash
# directory path problem 
cd /SHOW
ln -s $PWD/models ../models
ln -s $PWD/data ../data
ln -s $PWD/models/pymaf_data $PWD/modules/PyMAF/data
#ln -s $PWD/speech2gesture_dataset ../speech2gesture_dataset
mkdir /content
ln -s $PWD/openpose /content/openpose
ln -s $PWD/mmdetection /content/mmdetection
ln -s $PWD/mmdetection/mmpose /content/mmpose
```

## Fix torchgeometry bug

```bash
rm /miniconda3/lib/python3.9/site-packages/torchgeometry/core/conversions.py
cp /SHOW/conversions.py /miniconda3/lib/python3.9/site-packages/torchgeometry/core/conversions.py
```


</details>

# Test
In container, you might test the model:

```bash
cd /SHOW
python main.py --speaker_name -1 --all_top_dir ./test/demo_video/half.mp4
```




# TODO  
- Create a Docker image that requires no additional build steps or installations  
- Enable parallel processing support  
- Implement a host-container communication API (allowing video input and motion output)  
