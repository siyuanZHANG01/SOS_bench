# SOS_bench
Benchmarking various learned indexes on datasets of varying hardness and shifting workload

## Requirements

- **GCC**: 9.4.0+
- **CMake**: 3.14.0+

## Dependencies

- **Intel MKL**
- **Intel TBB**: 2020.1
- **jemalloc**
- **OpenMP**

> You can quickly set up the environment using the **Dockerfile** provided in this repository.

## Usage & Exploration

Please refer to the guide files in the following directories for running different components:

- **`hard_metric/`**: run hardness evaluation
- **`data/`**: run data generation
- **`src/`**: run performance benchmarking / testing

## Citation & Acknowledgements

This benchmark is developed based on **GRE**:
- Project: https://github.com/gre4index/GRE  
- Paper:  
  Chaichon Wongkham, Baotong Lu, Chris Liu, Zhicong Zhong, Eric Lo, and Tianzheng Wang.  
  *Are Updatable Learned Indexes Ready?* PVLDB, 15(11): 3004–3017, 2022.

We are also inspired by **RoBin**:
- https://github.com/cds-ruc/RoBin
