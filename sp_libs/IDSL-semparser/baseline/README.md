# IDSL-semparser-baseline


## Data Preparation
If the data is not extracted, you can do so by running the code below:
```bash
python pre-processing/extrac_raw.py
python pre-processing/raw2mrc.py
```

## Train model
Scripts for running our experiment can be found in the ./scripts/ folder. Note that you need to change DATA_DIR, BERT_DIR, OUTPUT_DIR to your own dataset path, bert model path and log path, respectively.

```bash
./scripts/semparser_baseline.sh
```
