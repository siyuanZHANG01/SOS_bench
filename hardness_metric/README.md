# Hardness Metric Evaluation

To evaluate the hardness of the data, please place the binary dataset files under a single directory, then run:

```bash
python3 main.py \
  --data_dir [your data dir] \
  --input_format bin \
  --mode pgm \
  --pgm_error_file [your .txt error file] \
  --out [fig_path, currently not available] \
  --csv_out [your csv path] \
  --jitter_duplicates
```

For `--pgm_error_file`, there should be **one line per data item**, and its value is set to `dataset_size / m`. We use `m = 3000`.

The evaluation results can be viewed in the output `.csv` file.

This evaluation program assumes that the dataset has **no duplicate keys**, therefore `--jitter_duplicates` must be enabled. It will perform a minimal upward adjustment for duplicate keys.

You can find the real datasets we used, or instructions for downloading them, at:
- `https://github.com/learnedsystems/SOSD`
- `https://github.com/gre4index/GRE`
