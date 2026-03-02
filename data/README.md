# Data Generation

You can generate datasets with different arrival rates and different hardness.

## 1) Generate timestamps (simulate different arrival rates)

`generate_ts_new.py` is used to produce a timestamp file based on the start time of each stable segment and the corresponding working-set size, in order to simulate different arrival rates.

If you need multiple stable segments, please provide an additional `events.json` file. Each entry corresponds to one stable segment:
- the first column/field corresponds to the start time of the stable period
- the second column/field corresponds to the working-set size at that time

If you also want to simulate distribution drift, you need to provide a `dis_events.json` file to specify the times at which you want distribution drift to occur.

Then, the program will output the timestamp file.

If you need to simulate distribution drift and you provide `dis_events.json`, the program will also output an auxiliary file to be used as the input of `data_generator.py`.

Example command:

```bash
python3 generate_ts_new.py \
  1000000000 \
  7500000000 \
  100000000 \
  events.json \
  --ts-file-type binary \
  --format lines \
  --out ts.bin \
  --dis-events dis_events.json \
  --dis-out num_axis.json \
  --trim-forced-tail
```

Positional arguments:
- `1000000000`: window length (`window_size`)
- `7500000000`: total time (`total_size`)
- `100000000`: initial working-set size (`initial_num`)

The last argument `--trim-forced-tail` is used to trim the occasionally extra timestamps.

## 2) Generate the key dataset

Then, use `data_generator.py` to generate the corresponding key dataset.

Four files are required:
- `mixture.json`: specifies the hyper-parameters used for generation
- `tau_top.json` and `tau_second.json`: specify the global/local hardness of each stable segment
- `num_axis.json`: determines the number of keys in each segment

Example command:

```bash
python3 data_generator.py \
  --mode Com \
  --mixture mixture.json \
  --time_axis num_axis.json \
  --tau_axis_top tau_top.json \
  --tau_axis_second tau_second.json \
  --outfile hehe_new.npy \
  --dtype uint64 \
  --avg_w_global 100000000
```

`Com` mode is the dedicated mode for generating distribution-drift data, and `avg_w_global` is the average working-set size you want.

If you do not want to simulate arrival-rate changes and only want to simulate distribution changes, you can provide `num_axis.json` yourself:
- on the i-th line/row, the first column is the number of keys in the i-th stable period
- the second column is the number of keys in the transition period
- the third column is currently overridden by the parameter `avg_w_global`

If you do not want to simulate distribution changes and only want to simulate arrival-rate changes, you can use `Div` mode and only provide:
- `mixture.json` (the hardness of the first distribution)
- `--total_num` (working-set size)
- `--gen_num` (total dataset size)

## 3) Dataset sizes used in the experiments

This section summarizes the dataset-size parameters used in our evaluations. In all cases below, set the **window size** equal to the **working-set length** when running the benchmark (i.e., use `--time_window` = working set size).

- **No distribution shift, no arrival-rate shift**:
  - Generate data by directly running `data_generator.py` in `Div` mode, with working-set size **100M** and total dataset size **500M**.

- **Distribution shift, no arrival-rate shift**:
  - Generate data by directly running `data_generator.py` in `Com` mode, with working-set size **100M** and total dataset size extended to **750M**, in order to cover the entire distribution-shift process.
  - Specify the number of keys in each stable period / transition period via `num_axis.json` (the current `num_axis.json` in this project corresponds to that configuration).

- **No distribution shift, arrival-rate shift**:
  - First, generate the timestamps using:

```bash
python3 generate_ts_new.py 1000000000 7000000000 100000000 events2.json --ts-file-type binary --format lines --out ts2.bin --dis-out test2.json --trim-forced-tail
```

  - Then, in `Div` mode, generate the key dataset with working-set size **100M** and total dataset size **500M**, and pass both the key dataset and the timestamp dataset to the evaluation program.

## 4) Notes

For follow-up, deeper development work on data distribution: `data_generator.py` is currently changed to accept **normalized MSE**, rather than the **unnormalized RMSE** defined in the paper. You can convert a hardness value(RMSE) into the normalized MSE by running:

```bash
python3 hardness_translation.py [your hardness] [local/global] [working set size] [m=3000]
```

Then, provide the converted normalized MSE in the JSON files used by the generator. In the experiment, lh=190 (passing 0.004 to the generator), le=13 (passing 2e-5), gh=92 (passing 0.004), and ge=7 (passing 2e-5).

If you don't want to generate the datasets for evaluation, you can get them here: https://drive.google.com/drive/folders/1KbViJCCT7D6MfzlPl0kBjDZeEYGLgdw2?usp=sharing

