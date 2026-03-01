# Evaluation (microbench)

To run the evaluation, you first need to provide a **key dataset** and the corresponding **timestamp dataset**.

If you do not want to simulate a varying arrival rate, then the timestamp dataset is **not required**. In that case, you can omit `--ts_file` (or pass it as empty), and provide the **working-set size** as the value of `--time_window`. The benchmark will automatically generate consecutive positive integers as timestamps.

Before the actual test, you need to run a **pre-run** to generate the operation-sequence file (`pre_run.bin`).

## 1) Pre-run (generate operation sequence)

Example command:

```bash
./microbench --streaming --pre_run \
  --keys_file=[your key file(s)] \
  --ts_file=[ts file or empty] \
  --keys_file_type=binary \
  --ts_file_type=binary \
  --table_size=500000000 \
  --time_window=1000000000 \
  --stream_mode=lookup \
  --rw_ratio=1.0 \
  --query_hit_ratio=1.0 \
  --query_distribution=uniform \
  --seed=1 \
  --shuffle_keys \
  --half_range \
  --stronger_non-duplicate
```

Notes:
- For data with distribution changes, do **not** set `--shuffle_keys`.
- To make `lipp` run correctly, enable `--stronger_non-duplicate` on high-hardness datasets.
- Enabling/disabling `--half_range` will not noticeably affect performance (it divides keys by 2 without changing the distribution). It exists for future adaptation to more indexes that are not evaluated in the paper.
- To avoid unexpected crashes, we recommend keeping `--query_hit_ratio=1.0` and `--query_distribution=uniform`, because various indexes have not been sufficiently evaluated under other workloads yet.

## 2) Single-thread test (streaming)

After generating the operation-sequence file, you can run the test.

Example command:

```bash
./microbench --streaming \
  --keys_file=[your key file(s)] \
  --ts_file=[ts file or empty] \
  --keys_file_type=binary \
  --ts_file_type=binary \
  --time_window=1000000000 \
  --stream_mode=lookup \
  --rw_ratio=1.0 \
  --query_hit_ratio=1.0 \
  --query_distribution=uniform \
  --seed=1 \
  --warm_up=1000000 \
  --latency_sample \
  --latency_sample_ratio=0.01 \
  --shuffle_keys \
  --segment_by=op \
  --segment_file=/app/SOS_EVA/build/segsize.txt \
  --index=artunsync,btree,imtree,lipp,pgm,sali,alex \
  --output_path=./out_streaming.txt \
  --pre_run_path=[gened_by_pre_run]
```

Explanation of key flags:
- `--warm_up`: number of operations used for warm-up.
- `--latency_sample` / `--latency_sample_ratio`: enable sampling (and sampling ratio) for percentile latency computation.
- `--segment_file`: a list of **space-separated positive integers**. When provided, the benchmark will collect statistics and output them at the specified timestamps / operation indices.
- `--segment_by`: selects whether the segment cuts are interpreted as timestamps (`ts`) or operation indices (`op`).

## 3) Multi-thread test

Multi-thread evaluation is similar. Example command:

```bash
./microbench --multithread \
  --keys_file=[your key file(s)] \
  --ts_file=[ts file or empty] \
  --keys_file_type=binary \
  --ts_file_type=binary \
  --time_window=1000000000 \
  --stream_mode=lookup \
  --rw_ratio=1.0 \
  --query_hit_ratio=1.0 \
  --query_distribution=uniform \
  --seed=1 \
  --shuffle_keys \
  --warm_up=1000000 \
  --latency_sample \
  --latency_sample_ratio=0.01 \
  --pre_run_path=[gened_by_pre_run] \
  --threads=48 \
  --round_ops=10000000000 \
  --dispatch=random \
  --index=${idx} \
  --output_path=./out_mullookup.txt
```

Notes:
- `--round_ops` forces threads to synchronize every fixed number of operations. At the current stage we do not recommend enabling it (please set it to at least **2000M**).
- `--dispatch` specifies how threads obtain operations. We recommend keeping it as `random`.

