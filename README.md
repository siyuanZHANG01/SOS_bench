# SOShift

## Requirements

- **GCC**: 9.4.0+
- **CMake**: 3.14.0+

## Dependencies

- **Intel MKL**
- **Intel TBB**: 2020.1
- **jemalloc**
- **OpenMP**
> You can quickly set up the environment using the **Dockerfile** provided in this repository.

## Build

build with CMake inside the container:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The benchmark binary will be at:

- **`build/microbench`**

## Usage & Exploration

Please refer to the guide files in the following directories for running different components:

- **`hard_metric/`**: run hardness evaluation
- **`data/`**: run data generation
- **`src/`**: run performance benchmarking / testing

This repository may be continuously updated.

This work utilizes some interfaces and tool files from [GRE](https://github.com/gre4index/GRE), as well as memory usage interfaces implemented by [Robin](https://github.com/cds-ruc/RoBin) for several indexes. We express our sincere gratitude to the authors.
