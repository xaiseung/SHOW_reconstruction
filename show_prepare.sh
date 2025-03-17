cd /SHOW

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

wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz && \
    tar -xzf cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-3.13.0-Linux-x86_64.tar.gz



# prevent 'dubious ownership' fatal error
git config --global --add safe.directory /SHOW/openpose
git config --global --add safe.directory /SHOW/openpose/3rdparty/caffe
git config --global --add safe.directory /SHOW/openpose/3rdparty/pybind11


cd /SHOW/openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     bash ~/miniconda.sh -bfp /miniconda3 && \
     rm ~/miniconda.sh && \
     eval "$(/miniconda3/bin/conda shell.bash hook)"
     
conda install -y python=3.9


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

# install pytorch3d
cd /SHOW
cd pytorch3d/
pip install -e .

# Resolve path mismatch issue
cd /SHOW
ln -s $PWD/models ../models
ln -s $PWD/data ../data
ln -s $PWD/models/pymaf_data $PWD/modules/PyMAF/data
#ln -s $PWD/speech2gesture_dataset ../speech2gesture_dataset
mkdir /content
ln -s $PWD/openpose /content/openpose
ln -s $PWD/mmdetection /content/mmdetection
ln -s $PWD/mmdetection/mmpose /content/mmpose


# Fix torchgeometry bug 
rm /miniconda3/lib/python3.9/site-packages/torchgeometry/core/conversions.py
cp /SHOW/conversions.py /miniconda3/lib/python3.9/site-packages/torchgeometry/core/conversions.py

