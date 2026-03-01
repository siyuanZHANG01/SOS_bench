## Prerequisites
This project uses [CMake](https://cmake.org/) (3.2+) and Ubuntu 18.04 for building and testing.
It also requires dependencies of [Intel MKL](https://software.intel.com/en-us/mkl), [jemalloc](https://github.com/jemalloc/jemalloc) and [userspace-rcu](https://github.com/urcu/userspace-rcu).

### Installing Intel MKL
Detailed steps can be found [here](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo).

```shell
$ cd /tmp
$ wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
$ apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
$ rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

$ sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
$ apt-get update
$ apt-get install -y intel-mkl-2020.4-912
```

After the installation, please modify the following two lines in `CMakeLists.txt` accordingly.

```cmake
set(MKL_LINK_DIRECTORY "/opt/intel/mkl/lib/intel64")
set(MKL_INCLUDE_DIRECTORY "/opt/intel/mkl/include")
```

### Installing jemalloc
```shell
$ apt-get -y install libjemalloc1
$ cd /usr/lib/x86_64-linux-gnu/
$ ln -s libjemalloc.so.1 libjemalloc.so
```

After the installation, please modify the following line in `CMakeLists.txt` accordingly. 

```cmake
set(JEMALLOC_DIR "/usr/lib/x86_64-linux-gnu")
```

### Installing urcu

```shell
$ git clone git://git.liburcu.org/userspace-rcu.git
$ ./bootstrap # skip if using tarball
$ ./configure
$ make
$ make install
$ ldconfig
```
After the installation, please modify the following line in `CMakeLists.txt` accordingly.

```cmake
include_directories("/home/userspace-rcu/include")
```

## Build and Run

We use cmake for LOFT 

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

To run the microbenchmark:

```shell
$ ./microbench
```

The [microbench](microbench.cpp) has several parameters you can pass, such as `read/insert`, `data_num` and configurations of LOFT.

```shell
$ ./microbench --bg_n 2 --fg_n 24 --data_num 100000000 --init_num 100000000 --oper_num 100000000 --benchmark 0 --insert 0 --read 1
```
## The orgnization of the code
The list of the files in the project:
```
├── bench
│   └── microbench.cpp  \\microbenchmark
├── CMakeLists.txt
├── data_node.h
├── data_node_impl.h
├── helper.h
├── LOFT.h
├── LOFT_impl.h
├── model.h
├── model_impl.h
├── piecewise_linear_model.hpp
├── README.md
├── root.h
├── root_impl.h
├── util.h
├── work_stealing.h
└── zipf.h
```
The relationship between the header files:
```
├── LOFT.h 
│   └── root.h  \\contains the strcuture of the root node and the background thread performing retraining and updating the model
│       └── data_node.h  \\contains the index operations upon data node and structure modification operations of data node
```
The impl files contain the implementation of the classes in the header files.
