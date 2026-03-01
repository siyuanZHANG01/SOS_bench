FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gnupg \
    software-properties-common \
    libjemalloc-dev \
    libboost-all-dev \
    libtbb-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && apt-get install -y \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV CPATH=$MKLROOT/include:$CPATH
ENV LIBRARY_PATH=$MKLROOT/lib/intel64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

WORKDIR /app

CMD ["/bin/bash"]