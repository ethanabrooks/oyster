# AUTHOR: Kristian Hartikainen
# Copied with minimal modifications from softlearning codebase:
#    https://github.com/rail-berkeley/softlearning
# MAINTAINER Kate Rakelly <rakelly@eecs.berkeley.edu>

# Base container that includes all dependencies but not the actual repo

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.1

#FROM nvidia/cudagl${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
FROM nvidia/cudagl:10.1-base-ubuntu18.04
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc
RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN conda update -y --name base conda \
    && conda clean --all -y

# ========== Repo dependencies ==========
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gnupg2 \
        make \
        cmake \
        ffmpeg \
        swig \
        libz-dev \
        unzip \
        zlib1g-dev \
        libglfw3 \
        libglfw3-dev \
        libxrandr2 \
        libxinerama-dev \
        libxi6 \
        libxcursor-dev \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        lsb-release \
        ack-grep \
        patchelf \
        vim \
        wget \
        xpra \
        xserver-xorg-dev \
        xvfb \
    && export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" \
    && echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
            | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | apt-key add - \
    && apt-get update -y \
    && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ========== Conda Environment ==========
COPY ./environment.yml /tmp/environment.yml

# NOTE: Don't separate the Mujoco key echo and remove commands into separate
# run commands! Otherwise your key will be readable by anyone who has access
# To the container. We need the key in order to compile mujoco_py.
# TODO this is not good but it can't find the key otherwise
#RUN echo "${MJKEY}" > ./mjkey.txt 
RUN conda env update -f /tmp/environment.yml \
    && conda clean --all -y 

RUN mkdir /root/code
WORKDIR /root/code

# add rand_param_envs to python path
ENV PYTHONPATH="rand_param_envs:$PYTHONPATH"
RUN echo "source activate pearl" >> /root/.bashrc \
    && echo "cd /root/code" >> /root/.bashrc
ENV PATH /opt/conda/envs/pearl/bin:$PATH

COPY . /root/code
ENTRYPOINT ["/root/code/entrypoint.sh"]
