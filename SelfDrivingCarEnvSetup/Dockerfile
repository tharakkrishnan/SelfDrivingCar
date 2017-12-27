FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

MAINTAINER @joshuacook

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
	python \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
	tcl-dev \
	tk-dev \
	libcupti-dev \	
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh tmp/Miniconda3-4.2.12-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.2.12-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment-gpu.yml  ./environment.yml
RUN conda env create -f=environment.yml --name carnd-term1 --debug -v -v

# cleanup tarballs and downloaded package files
RUN conda clean -tp -y

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow GPU version.
ENV TENSORFLOW_VERSION 0.6.0 
RUN curl -O https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
RUN pip --no-cache-dir install \
    tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

#RUN pip install --upgrade tensorflow-gpu

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Term 1 workdir
RUN mkdir /src
WORKDIR "/src"

# Make sure CUDNN is detected
ENV LD_LIBRARY_PATH  /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64

# Copying local version which works with tensorflow
COPY libcudnn.so.5 /usr/local/cuda/lib64/
RUN ln -s /usr/local/cuda/lib64/libcudnn.so.5 /usr/local/cuda/lib64/libcudnn.so

# TensorBoard
EXPOSE 6006
# Jupyter
EXPOSE 8888
# Flask Server
EXPOSE 4567

## Two Birds, One Stone
# 1. sources conda environment
# 2. prevents the zombie container issue when started as pid 1
COPY run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
